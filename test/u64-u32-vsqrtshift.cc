// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/u64-u32-vsqrtshift.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


TEST(U64_U32_VSQRTSHIFT__SCALAR_CVTU32_SQRT_CVTU32F64_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_u64_u32_vsqrtshift_ukernel__scalar_cvtu32_sqrt_cvtu32f64_u1);
}

TEST(U64_U32_VSQRTSHIFT__SCALAR_CVTU32_SQRT_CVTU32F64_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u64_u32_vsqrtshift_ukernel__scalar_cvtu32_sqrt_cvtu32f64_u1);
  }
}

TEST(U64_U32_VSQRTSHIFT__SCALAR_CVTU32_SQRT_CVTU32F64_U1, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .shift(shift)
        .Test(xnn_u64_u32_vsqrtshift_ukernel__scalar_cvtu32_sqrt_cvtu32f64_u1);
    }
  }
}