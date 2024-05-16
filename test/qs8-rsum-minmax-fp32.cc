// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-rsum-minmax-fp32.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>
#include "rsum-microkernel-tester.h"


TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U1, batch_eq_1) {
  RSumMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U1, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(2)
      .scale(scale)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U1, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(32)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, batch_eq_2) {
  RSumMicrokernelTester()
    .batch_size(2)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(3)
      .scale(scale)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U2, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(64)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, batch_eq_4) {
  RSumMicrokernelTester()
    .batch_size(4)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(5)
      .scale(scale)
      .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RSUM_MINMAX_FP32__SCALAR_IMAGIC_U4, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(128)
    .Test(xnn_qs8_rsum_minmax_fp32_ukernel__scalar_imagic_u4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}