// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vlog.yaml
//   Generator: tools/generate-vunary-test.py


#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>

#include <gtest/gtest.h>
#include "vunary-microkernel-tester.h"


TEST(F32_VLOG__SCALAR_LOG_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .TestLog(xnn_f32_vlog_ukernel__scalar_log_u1);
}

TEST(F32_VLOG__SCALAR_LOG_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u1);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u1);
  }
}


TEST(F32_VLOG__SCALAR_LOG_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
}

TEST(F32_VLOG__SCALAR_LOG_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
  }
}


TEST(F32_VLOG__SCALAR_LOG_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
}

TEST(F32_VLOG__SCALAR_LOG_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
  }
}
