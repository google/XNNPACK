// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/u32-vlog.yaml
//   Generator: tools/generate-vlog-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vlog.h>
#include "vlog-microkernel-tester.h"


TEST(U32_VLOG__SCALAR_X1, DISABLED_batch_eq_1) {
  VLogMicrokernelTester()
    .batch(1)
    .Test(xnn_u32_vlog_ukernel__scalar_x1);
}

TEST(U32_VLOG__SCALAR_X1, DISABLED_batch_gt_1) {
  for (size_t batch = 2; batch < 10; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x1);
  }
}

TEST(U32_VLOG__SCALAR_X1, DISABLED_input_lshift) {
  for (uint32_t input_lshift = 0; input_lshift < 32; input_lshift++) {
    VLogMicrokernelTester()
      .batch(1)
      .input_lshift(input_lshift)
      .Test(xnn_u32_vlog_ukernel__scalar_x1);
  }
}

TEST(U32_VLOG__SCALAR_X1, DISABLED_output_scale) {
  for (uint32_t output_scale = 0; output_scale < 65536; output_scale += 3) {
    VLogMicrokernelTester()
      .batch(1)
      .output_scale(output_scale)
      .Test(xnn_u32_vlog_ukernel__scalar_x1);
  }
}

TEST(U32_VLOG__SCALAR_X1, DISABLED_inplace) {
  for (size_t batch = 2; batch < 10; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .Test(xnn_u32_vlog_ukernel__scalar_x1);
  }
}


TEST(U32_VLOG__SCALAR_X2, DISABLED_batch_eq_2) {
  VLogMicrokernelTester()
    .batch(2)
    .Test(xnn_u32_vlog_ukernel__scalar_x2);
}

TEST(U32_VLOG__SCALAR_X2, DISABLED_batch_div_2) {
  for (size_t batch = 4; batch < 20; batch += 2) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x2);
  }
}

TEST(U32_VLOG__SCALAR_X2, DISABLED_batch_lt_2) {
  for (size_t batch = 1; batch < 2; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x2);
  }
}

TEST(U32_VLOG__SCALAR_X2, DISABLED_batch_gt_2) {
  for (size_t batch = 3; batch < 4; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x2);
  }
}

TEST(U32_VLOG__SCALAR_X2, DISABLED_input_lshift) {
  for (uint32_t input_lshift = 0; input_lshift < 32; input_lshift++) {
    VLogMicrokernelTester()
      .batch(2)
      .input_lshift(input_lshift)
      .Test(xnn_u32_vlog_ukernel__scalar_x2);
  }
}

TEST(U32_VLOG__SCALAR_X2, DISABLED_output_scale) {
  for (uint32_t output_scale = 0; output_scale < 65536; output_scale += 5) {
    VLogMicrokernelTester()
      .batch(2)
      .output_scale(output_scale)
      .Test(xnn_u32_vlog_ukernel__scalar_x2);
  }
}

TEST(U32_VLOG__SCALAR_X2, DISABLED_inplace) {
  for (size_t batch = 3; batch < 4; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .Test(xnn_u32_vlog_ukernel__scalar_x2);
  }
}


TEST(U32_VLOG__SCALAR_X3, DISABLED_batch_eq_3) {
  VLogMicrokernelTester()
    .batch(3)
    .Test(xnn_u32_vlog_ukernel__scalar_x3);
}

TEST(U32_VLOG__SCALAR_X3, DISABLED_batch_div_3) {
  for (size_t batch = 6; batch < 30; batch += 3) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x3);
  }
}

TEST(U32_VLOG__SCALAR_X3, DISABLED_batch_lt_3) {
  for (size_t batch = 1; batch < 3; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x3);
  }
}

TEST(U32_VLOG__SCALAR_X3, DISABLED_batch_gt_3) {
  for (size_t batch = 4; batch < 6; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x3);
  }
}

TEST(U32_VLOG__SCALAR_X3, DISABLED_input_lshift) {
  for (uint32_t input_lshift = 0; input_lshift < 32; input_lshift++) {
    VLogMicrokernelTester()
      .batch(3)
      .input_lshift(input_lshift)
      .Test(xnn_u32_vlog_ukernel__scalar_x3);
  }
}

TEST(U32_VLOG__SCALAR_X3, DISABLED_output_scale) {
  for (uint32_t output_scale = 0; output_scale < 65536; output_scale += 5) {
    VLogMicrokernelTester()
      .batch(3)
      .output_scale(output_scale)
      .Test(xnn_u32_vlog_ukernel__scalar_x3);
  }
}

TEST(U32_VLOG__SCALAR_X3, DISABLED_inplace) {
  for (size_t batch = 4; batch < 6; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .Test(xnn_u32_vlog_ukernel__scalar_x3);
  }
}


TEST(U32_VLOG__SCALAR_X4, DISABLED_batch_eq_4) {
  VLogMicrokernelTester()
    .batch(4)
    .Test(xnn_u32_vlog_ukernel__scalar_x4);
}

TEST(U32_VLOG__SCALAR_X4, DISABLED_batch_div_4) {
  for (size_t batch = 8; batch < 40; batch += 4) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x4);
  }
}

TEST(U32_VLOG__SCALAR_X4, DISABLED_batch_lt_4) {
  for (size_t batch = 1; batch < 4; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x4);
  }
}

TEST(U32_VLOG__SCALAR_X4, DISABLED_batch_gt_4) {
  for (size_t batch = 5; batch < 8; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_vlog_ukernel__scalar_x4);
  }
}

TEST(U32_VLOG__SCALAR_X4, DISABLED_input_lshift) {
  for (uint32_t input_lshift = 0; input_lshift < 32; input_lshift++) {
    VLogMicrokernelTester()
      .batch(4)
      .input_lshift(input_lshift)
      .Test(xnn_u32_vlog_ukernel__scalar_x4);
  }
}

TEST(U32_VLOG__SCALAR_X4, DISABLED_output_scale) {
  for (uint32_t output_scale = 0; output_scale < 65536; output_scale += 7) {
    VLogMicrokernelTester()
      .batch(4)
      .output_scale(output_scale)
      .Test(xnn_u32_vlog_ukernel__scalar_x4);
  }
}

TEST(U32_VLOG__SCALAR_X4, DISABLED_inplace) {
  for (size_t batch = 5; batch < 8; batch++) {
    VLogMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .Test(xnn_u32_vlog_ukernel__scalar_x4);
  }
}
