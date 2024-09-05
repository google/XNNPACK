// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rwdsum.yaml
//   Generator: tools/generate-rwd-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "rwd-microkernel-tester.h"


TEST(F32_RWDSUM_1P1X__SCALAR_C1, channels_eq_1_2pass_fulltile) {
  RWDMicrokernelTester()
    .rows(2)
    .channels(1)
    .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
}

TEST(F32_RWDSUM_1P1X__SCALAR_C1, channels_eq_1_2pass_subtile) {
  for (size_t rows = 1; rows < 2; rows++) {
    RWDMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
  }
}

TEST(F32_RWDSUM_1P1X__SCALAR_C1, channels_eq_1_multipass_fulltile) {
  for (size_t rows = 1; rows <= 5; rows += 1) {
    RWDMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
  }
}

TEST(F32_RWDSUM_1P1X__SCALAR_C1, channels_div_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    RWDMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
  }
}

TEST(F32_RWDSUM_1P1X__SCALAR_C1, channels_div_1_2pass_subtile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 1; rows < 2; rows++) {
      RWDMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
    }
  }
}

TEST(F32_RWDSUM_1P1X__SCALAR_C1, channels_div_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 1; rows <= 5; rows += 1) {
      RWDMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
    }
  }
}

TEST(F32_RWDSUM_1P1X__SCALAR_C1, channels_gt_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    RWDMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
  }
}

TEST(F32_RWDSUM_1P1X__SCALAR_C1, channels_gt_1_2pass_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 1; rows < 2; rows++) {
      RWDMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
    }
  }
}

TEST(F32_RWDSUM_1P1X__SCALAR_C1, channels_gt_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 1; rows < 5; rows += 2) {
      RWDMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
    }
  }
}

TEST(F32_RWDSUM_1P1X__SCALAR_C1, overflow_accumulator) {
  for (size_t channels = 1; channels < 2; ++channels) {
     RWDMicrokernelTester()
       .rows(258)
       .channels(channels)
       .Test(xnn_f32_rwdsum_ukernel_1p1x__scalar_c1, RWDMicrokernelTester::OpType::Sum);
  }
}