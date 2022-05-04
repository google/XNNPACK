// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-gavgpool-minmax-fp32.yaml
//   Generator: tools/generate-gavgpool-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_eq_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_eq_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_lt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_lt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_gt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_gt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_eq_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_eq_24_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_eq_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_eq_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_eq_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_eq_24_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_eq_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_eq_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_div_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_div_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_div_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_div_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_lt_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_lt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_lt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_lt_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_lt_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_lt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_gt_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_gt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_gt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_gt_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_gt_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C24, channels_gt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_eq_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_eq_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_lt_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_lt_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_gt_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_gt_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEON_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_eq_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_eq_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_lt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_lt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_gt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_gt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_eq_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_eq_24_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_eq_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_eq_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_eq_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_eq_24_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_eq_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_eq_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_div_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_div_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_div_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_div_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_lt_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_lt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_lt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_lt_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_lt_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_lt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_gt_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_gt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_gt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_gt_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_gt_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C24, channels_gt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_eq_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_eq_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_lt_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_lt_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_gt_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_gt_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__NEONV8_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c8, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_eq_16_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_eq_16_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_eq_16_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_eq_16_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_div_16_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_lt_16_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_lt_16_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_lt_16_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_gt_16_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_gt_16_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C16, channels_gt_16_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c16, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_eq_24_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_eq_24_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_eq_24_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_eq_24_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_eq_24_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_div_24_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_div_24_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_lt_24_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_lt_24_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_lt_24_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_lt_24_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_gt_24_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_gt_24_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_gt_24_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C24, channels_gt_24_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c24, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_eq_32_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_eq_32_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_eq_32_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .input_stride(37)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_eq_32_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_eq_32_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_div_32_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_div_32_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_lt_32_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_lt_32_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_lt_32_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_lt_32_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_gt_32_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_gt_32_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_gt_32_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEON_C32, channels_gt_32_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neon_c32, xnn_init_qs8_avgpool_minmax_fp32_neon_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c8, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_eq_16_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_eq_16_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_eq_16_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_eq_16_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_div_16_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_lt_16_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_lt_16_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_lt_16_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_gt_16_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_gt_16_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C16, channels_gt_16_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c16, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_eq_24_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_eq_24_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_eq_24_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_eq_24_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_eq_24_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_div_24_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_div_24_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_lt_24_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_lt_24_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_lt_24_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_lt_24_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_gt_24_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_gt_24_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_gt_24_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C24, channels_gt_24_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c24, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_eq_32_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_eq_32_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_eq_32_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .input_stride(37)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_eq_32_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_eq_32_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_div_32_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_div_32_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_lt_32_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_lt_32_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_lt_32_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_lt_32_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_gt_32_fulltile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_gt_32_subtile) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_gt_32_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__NEONV8_C32, channels_gt_32_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__neonv8_c32, xnn_init_qs8_avgpool_minmax_fp32_neonv8_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_eq_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_eq_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_lt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_lt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_gt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_gt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_eq_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_eq_24_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_eq_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_eq_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_eq_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_eq_24_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_eq_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_eq_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_div_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_div_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_div_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_div_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_lt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_lt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_lt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_lt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_lt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_lt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_gt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_gt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_gt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_gt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_gt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE2_C24, channels_gt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_eq_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_eq_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_lt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_lt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_gt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_gt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_eq_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_eq_24_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_eq_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_eq_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_eq_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_eq_24_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_eq_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_eq_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_div_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_div_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_div_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_div_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_lt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_lt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_lt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_lt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_lt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_lt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_gt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_gt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_gt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_gt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_gt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SSE41_C24, channels_gt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_eq_16_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_eq_16_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_eq_16_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_eq_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_eq_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_div_16_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_div_16_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_lt_16_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_lt_16_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_lt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_lt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_gt_16_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_gt_16_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_gt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C16, channels_gt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c16, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_eq_24_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_eq_24_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_eq_24_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_eq_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_eq_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_div_24_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_div_24_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_lt_24_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_lt_24_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_lt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_lt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_gt_24_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_gt_24_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_gt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE2_C24, channels_gt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c24, xnn_init_qs8_avgpool_minmax_fp32_sse2_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c8, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_eq_16_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_eq_16_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_eq_16_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_eq_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_eq_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_div_16_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_div_16_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_lt_16_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_lt_16_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_lt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_lt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_gt_16_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_gt_16_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_gt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C16, channels_gt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c16, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_eq_24_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_eq_24_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_eq_24_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_eq_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_eq_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_div_24_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_div_24_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_lt_24_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_lt_24_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_lt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_lt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_gt_24_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_gt_24_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_gt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SSE41_C24, channels_gt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse41_c24, xnn_init_qs8_avgpool_minmax_fp32_sse4_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_eq_8_2pass_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_eq_8_2pass_subtile) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_eq_8_multipass_fulltile) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_div_8_2pass_fulltile) {
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_div_8_2pass_subtile) {
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_div_8_multipass_fulltile) {
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_lt_8_2pass_fulltile) {
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_lt_8_2pass_subtile) {
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_lt_8_multipass_fulltile) {
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_gt_8_2pass_fulltile) {
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_gt_8_2pass_subtile) {
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_gt_8_multipass_fulltile) {
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_eq_16_2pass_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_eq_16_2pass_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_eq_16_2pass_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_eq_16_2pass_subtile) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_eq_16_multipass_fulltile) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_div_16_2pass_fulltile) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_div_16_2pass_subtile) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_div_16_multipass_fulltile) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_lt_16_2pass_fulltile) {
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_lt_16_2pass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_lt_16_2pass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_lt_16_2pass_subtile) {
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_lt_16_multipass_fulltile) {
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_gt_16_2pass_fulltile) {
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_gt_16_2pass_fulltile_with_qmax) {
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_gt_16_2pass_fulltile_with_qmin) {
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_gt_16_2pass_subtile) {
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_gt_16_multipass_fulltile) {
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_eq_24_2pass_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_eq_24_2pass_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_eq_24_2pass_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_eq_24_2pass_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_eq_24_2pass_subtile) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_eq_24_2pass_subtile_with_input_stride) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_eq_24_multipass_fulltile) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_eq_24_multipass_fulltile_with_input_stride) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_div_24_2pass_fulltile) {
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_div_24_2pass_subtile) {
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_div_24_multipass_fulltile) {
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_div_24_multipass_fulltile_with_input_stride) {
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_lt_24_2pass_fulltile) {
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_lt_24_2pass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_lt_24_2pass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_lt_24_2pass_subtile) {
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_lt_24_multipass_fulltile) {
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_lt_24_multipass_fulltile_with_input_stride) {
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_gt_24_2pass_fulltile) {
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_gt_24_2pass_fulltile_with_qmax) {
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_gt_24_2pass_fulltile_with_qmin) {
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_gt_24_2pass_subtile) {
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_gt_24_multipass_fulltile) {
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C24, channels_gt_24_multipass_fulltile_with_input_stride) {
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_eq_32_2pass_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_eq_32_2pass_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_eq_32_2pass_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_eq_32_2pass_subtile) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_eq_32_multipass_fulltile) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_div_32_2pass_fulltile) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_div_32_2pass_subtile) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_div_32_multipass_fulltile) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_lt_32_2pass_fulltile) {
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_lt_32_2pass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_lt_32_2pass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_lt_32_2pass_subtile) {
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_lt_32_multipass_fulltile) {
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_gt_32_2pass_fulltile) {
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_gt_32_2pass_fulltile_with_qmax) {
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_gt_32_2pass_fulltile_with_qmin) {
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_gt_32_2pass_subtile) {
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_gt_32_multipass_fulltile) {
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__WASMSIMD_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_eq_8_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_eq_8_subtile) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_eq_8_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_eq_8_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_eq_8_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_div_8_fulltile) {
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_div_8_subtile) {
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_lt_8_fulltile) {
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_lt_8_subtile) {
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_lt_8_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_lt_8_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_gt_8_fulltile) {
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_gt_8_subtile) {
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_gt_8_fulltile_with_qmax) {
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C8, channels_gt_8_fulltile_with_qmin) {
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_eq_16_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_eq_16_subtile) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_eq_16_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_eq_16_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_eq_16_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_div_16_fulltile) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_div_16_subtile) {
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_lt_16_fulltile) {
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_lt_16_subtile) {
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_lt_16_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_lt_16_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_gt_16_fulltile) {
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_gt_16_subtile) {
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_gt_16_fulltile_with_qmax) {
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C16, channels_gt_16_fulltile_with_qmin) {
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c16, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_eq_24_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_eq_24_subtile) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_eq_24_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_eq_24_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_eq_24_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_div_24_fulltile) {
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_div_24_subtile) {
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_lt_24_fulltile) {
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_lt_24_subtile) {
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_lt_24_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_lt_24_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_gt_24_fulltile) {
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_gt_24_subtile) {
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_gt_24_fulltile_with_qmax) {
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C24, channels_gt_24_fulltile_with_qmin) {
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c24, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_eq_32_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_eq_32_subtile) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_eq_32_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .input_stride(37)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_eq_32_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_eq_32_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_div_32_fulltile) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_div_32_subtile) {
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_lt_32_fulltile) {
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_lt_32_subtile) {
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_lt_32_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_lt_32_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_gt_32_fulltile) {
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_gt_32_subtile) {
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_gt_32_fulltile_with_qmax) {
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__WASMSIMD_C32, channels_gt_32_fulltile_with_qmin) {
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32, xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_eq_1_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_eq_1_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .input_stride(3)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_eq_1_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_eq_1_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_eq_1_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_eq_1_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_eq_1_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_eq_1_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_div_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_div_1_2pass_subtile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_div_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_div_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_gt_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_gt_1_2pass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_gt_1_2pass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_gt_1_2pass_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_gt_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C1, channels_gt_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(17)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}


TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_eq_2_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_eq_2_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .input_stride(5)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_eq_2_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_eq_2_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_eq_2_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_eq_2_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .input_stride(5)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_eq_2_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_eq_2_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .input_stride(5)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_div_2_2pass_fulltile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_div_2_2pass_subtile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_div_2_multipass_fulltile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_div_2_multipass_fulltile_with_input_stride) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(37)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_lt_2_2pass_fulltile) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_lt_2_2pass_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_lt_2_2pass_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_lt_2_2pass_subtile) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_lt_2_multipass_fulltile) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_lt_2_multipass_fulltile_with_input_stride) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(5)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_gt_2_2pass_fulltile) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_gt_2_2pass_fulltile_with_qmax) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_gt_2_2pass_fulltile_with_qmin) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_gt_2_2pass_subtile) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_gt_2_multipass_fulltile) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C2, channels_gt_2_multipass_fulltile_with_input_stride) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(17)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}


TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_eq_4_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .input_stride(7)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_eq_4_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_eq_4_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_eq_4_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_eq_4_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_eq_4_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_div_4_2pass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_div_4_2pass_subtile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_div_4_multipass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_div_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(67)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_lt_4_2pass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_lt_4_2pass_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_lt_4_2pass_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_lt_4_2pass_subtile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_lt_4_multipass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(7)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_gt_4_2pass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_gt_4_2pass_fulltile_with_qmax) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_gt_4_2pass_fulltile_with_qmin) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_gt_4_2pass_subtile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_gt_4_multipass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_FMAGIC_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}


TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_eq_1_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_eq_1_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .input_stride(3)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_eq_1_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_eq_1_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_eq_1_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_eq_1_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_div_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_div_1_2pass_subtile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_div_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_div_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_gt_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_gt_1_2pass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_gt_1_2pass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_gt_1_2pass_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(17)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}


TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_eq_2_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_eq_2_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .input_stride(5)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_eq_2_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_eq_2_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_eq_2_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_eq_2_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .input_stride(5)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_eq_2_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_eq_2_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .input_stride(5)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_div_2_2pass_fulltile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_div_2_2pass_subtile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_div_2_multipass_fulltile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_div_2_multipass_fulltile_with_input_stride) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(37)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_lt_2_2pass_fulltile) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_lt_2_2pass_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_lt_2_2pass_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_lt_2_2pass_subtile) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_lt_2_multipass_fulltile) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_lt_2_multipass_fulltile_with_input_stride) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(5)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_gt_2_2pass_fulltile) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_gt_2_2pass_fulltile_with_qmax) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_gt_2_2pass_fulltile_with_qmin) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_gt_2_2pass_subtile) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_gt_2_multipass_fulltile) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C2, channels_gt_2_multipass_fulltile_with_input_stride) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(17)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}


TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_eq_4_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .input_stride(7)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_eq_4_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_eq_4_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_eq_4_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_eq_4_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_eq_4_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_div_4_2pass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_div_4_2pass_subtile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_div_4_multipass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_div_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(67)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_lt_4_2pass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_lt_4_2pass_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_lt_4_2pass_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_lt_4_2pass_subtile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_lt_4_multipass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(7)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_gt_4_2pass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_gt_4_2pass_fulltile_with_qmax) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_gt_4_2pass_fulltile_with_qmin) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_gt_4_2pass_subtile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_gt_4_multipass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_IMAGIC_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}


TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_eq_1_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_eq_1_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .input_stride(3)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_eq_1_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_eq_1_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_eq_1_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_eq_1_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_eq_1_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_eq_1_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_div_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_div_1_2pass_subtile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_div_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_div_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_gt_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_gt_1_2pass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_gt_1_2pass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_gt_1_2pass_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_gt_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C1, channels_gt_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(17)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}


TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_eq_2_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_eq_2_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .input_stride(5)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_eq_2_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_eq_2_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(2)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_eq_2_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_eq_2_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .input_stride(5)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_eq_2_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_eq_2_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .input_stride(5)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_div_2_2pass_fulltile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_div_2_2pass_subtile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_div_2_multipass_fulltile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_div_2_multipass_fulltile_with_input_stride) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(37)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_lt_2_2pass_fulltile) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_lt_2_2pass_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_lt_2_2pass_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_lt_2_2pass_subtile) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_lt_2_multipass_fulltile) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_lt_2_multipass_fulltile_with_input_stride) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(5)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_gt_2_2pass_fulltile) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_gt_2_2pass_fulltile_with_qmax) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_gt_2_2pass_fulltile_with_qmin) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_gt_2_2pass_subtile) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_gt_2_multipass_fulltile) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C2, channels_gt_2_multipass_fulltile_with_input_stride) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(17)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}


TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_eq_4_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .input_stride(7)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_eq_4_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_eq_4_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(4)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_eq_4_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_eq_4_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_eq_4_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .input_stride(7)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_div_4_2pass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_div_4_2pass_subtile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_div_4_multipass_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_div_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(67)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_lt_4_2pass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_lt_4_2pass_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_lt_4_2pass_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_lt_4_2pass_subtile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_lt_4_multipass_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(7)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_gt_4_2pass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_gt_4_2pass_fulltile_with_qmax) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_gt_4_2pass_fulltile_with_qmin) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_gt_4_2pass_subtile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_gt_4_multipass_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7P7X__SCALAR_LRINTF_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}


TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C1, channels_eq_1_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C1, channels_eq_1_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C1, channels_eq_1_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .input_stride(3)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C1, channels_eq_1_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C1, channels_eq_1_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C1, channels_gt_1_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C1, channels_gt_1_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C1, channels_gt_1_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C1, channels_gt_1_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_eq_2_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_eq_2_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_eq_2_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .input_stride(5)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_eq_2_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_eq_2_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_div_2_fulltile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_div_2_subtile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_lt_2_fulltile) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_lt_2_subtile) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_lt_2_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_lt_2_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_gt_2_fulltile) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_gt_2_subtile) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_gt_2_fulltile_with_qmax) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C2, channels_gt_2_fulltile_with_qmin) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_eq_4_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_eq_4_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_eq_4_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .input_stride(7)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_eq_4_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_eq_4_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_div_4_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_div_4_subtile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_lt_4_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_lt_4_subtile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_lt_4_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_lt_4_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_gt_4_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_gt_4_subtile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_gt_4_fulltile_with_qmax) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_FMAGIC_C4, channels_gt_4_fulltile_with_qmin) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_fmagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C1, channels_eq_1_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C1, channels_eq_1_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C1, channels_eq_1_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .input_stride(3)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C1, channels_eq_1_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C1, channels_eq_1_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C1, channels_gt_1_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C1, channels_gt_1_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C1, channels_gt_1_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C1, channels_gt_1_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_eq_2_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_eq_2_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_eq_2_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .input_stride(5)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_eq_2_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_eq_2_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_div_2_fulltile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_div_2_subtile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_lt_2_fulltile) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_lt_2_subtile) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_lt_2_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_lt_2_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_gt_2_fulltile) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_gt_2_subtile) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_gt_2_fulltile_with_qmax) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C2, channels_gt_2_fulltile_with_qmin) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_eq_4_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_eq_4_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_eq_4_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .input_stride(7)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_eq_4_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_eq_4_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_div_4_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_div_4_subtile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_lt_4_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_lt_4_subtile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_lt_4_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_lt_4_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_gt_4_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_gt_4_subtile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_gt_4_fulltile_with_qmax) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_IMAGIC_C4, channels_gt_4_fulltile_with_qmin) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_imagic_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C1, channels_eq_1_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C1, channels_eq_1_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C1, channels_eq_1_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .input_stride(3)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C1, channels_eq_1_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C1, channels_eq_1_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C1, channels_gt_1_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C1, channels_gt_1_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C1, channels_gt_1_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C1, channels_gt_1_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c1, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_eq_2_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_eq_2_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(2)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_eq_2_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .input_stride(5)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_eq_2_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_eq_2_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(2)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_div_2_fulltile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_div_2_subtile) {
  for (size_t channels = 4; channels < 16; channels += 2) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_lt_2_fulltile) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_lt_2_subtile) {
  for (size_t channels = 1; channels < 2; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_lt_2_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_lt_2_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 2; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_gt_2_fulltile) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_gt_2_subtile) {
  for (size_t channels = 3; channels < 4; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_gt_2_fulltile_with_qmax) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C2, channels_gt_2_fulltile_with_qmin) {
  for (size_t channels = 3; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c2, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_eq_4_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_eq_4_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(4)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_eq_4_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .input_stride(7)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_eq_4_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .qmax(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_eq_4_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(4)
    .qmin(128)
    .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_div_4_fulltile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_div_4_subtile) {
  for (size_t channels = 8; channels < 32; channels += 4) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_lt_4_fulltile) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_lt_4_subtile) {
  for (size_t channels = 1; channels < 4; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_lt_4_fulltile_with_qmax) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_lt_4_fulltile_with_qmin) {
  for (size_t channels = 1; channels < 4; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_gt_4_fulltile) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_gt_4_subtile) {
  for (size_t channels = 5; channels < 8; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_gt_4_fulltile_with_qmax) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_GAVGPOOL_MINMAX_FP32_7X__SCALAR_LRINTF_C4, channels_gt_4_fulltile_with_qmin) {
  for (size_t channels = 5; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__scalar_lrintf_c4, xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params, xnn_qs8_requantize_fp32);
  }
}