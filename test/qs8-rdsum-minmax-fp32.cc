// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-rdsum-minmax-fp32.yaml
//   Generator: tools/generate-rdsum-test.py


#include <xnnpack/common.h>
#include <xnnpack/reduce.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>

#include "rdsum-microkernel-tester.h"
#include <gtest/gtest.h>


TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_eq_4_2pass_fulltile) {
  RDSumMicrokernelTester()
    .rows(14)
    .channels(4)
    .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
  RDSumMicrokernelTester()
    .rows(14)
    .channels(4)
    .input_stride(7)
    .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_eq_4_2pass_subtile) {
  for (size_t rows = 1; rows < 14; rows++) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_eq_4_2pass_subtile_with_input_stride) {
  for (size_t rows = 1; rows < 14; rows++) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_eq_4_multipass_fulltile) {
  for (size_t rows = 1; rows <= 35; rows += 7) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
  for (size_t rows = 1; rows <= 35; rows += 7) {
    RDSumMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_div_4_2pass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_div_4_2pass_subtile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_div_4_multipass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_div_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(67)
        .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_lt_4_2pass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_lt_4_2pass_subtile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_lt_4_multipass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 1; rows <= 35; rows += 7) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(7)
        .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_gt_4_2pass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    RDSumMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_gt_4_2pass_subtile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 14; rows++) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_gt_4_multipass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 35; rows += 14) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_RDSUM_MINMAX_FP32_7P7X__SCALAR_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 35; rows += 14) {
      RDSumMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .Test(xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}