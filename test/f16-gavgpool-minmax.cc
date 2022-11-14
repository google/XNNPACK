// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-gavgpool-minmax.yaml
//   Generator: tools/generate-gavgpool-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_lt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_lt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_gt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_gt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_eq_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_eq_24_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_eq_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_eq_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_eq_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_eq_24_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_eq_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_eq_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_div_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_div_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_div_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_div_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_lt_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_lt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_lt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_lt_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_lt_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_lt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_gt_24_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_gt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_gt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_gt_24_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_gt_24_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C24, channels_gt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_lt_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_lt_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_gt_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_gt_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_eq_16_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_eq_16_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_eq_16_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_eq_16_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_div_16_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_lt_16_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_lt_16_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_lt_16_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_gt_16_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_gt_16_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C16, channels_gt_16_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c16, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_eq_24_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_eq_24_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_eq_24_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_eq_24_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_eq_24_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_div_24_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_div_24_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_lt_24_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_lt_24_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_lt_24_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_lt_24_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_gt_24_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_gt_24_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_gt_24_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C24, channels_gt_24_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c24, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_eq_32_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_eq_32_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_eq_32_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .input_stride(37)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_eq_32_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_eq_32_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_div_32_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_div_32_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_lt_32_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_lt_32_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_lt_32_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_lt_32_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_gt_32_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_gt_32_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_gt_32_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C32, channels_gt_32_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_eq_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_eq_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_lt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_lt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_gt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_gt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C16, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_eq_24_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_eq_24_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_eq_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_eq_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_eq_24_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_eq_24_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_eq_24_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_eq_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_div_24_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_div_24_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_div_24_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_div_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_lt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_lt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_lt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_lt_24_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_lt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_lt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_gt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_gt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_gt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_gt_24_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_gt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C24, channels_gt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_eq_32_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_eq_32_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .input_stride(37)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_eq_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_eq_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(32)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_eq_32_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_eq_32_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_eq_32_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_eq_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .input_stride(37)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_div_32_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_div_32_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_div_32_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_div_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(521)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_lt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_lt_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_lt_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_lt_32_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_lt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_lt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(37)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_gt_32_2pass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_gt_32_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_gt_32_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_gt_32_2pass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_gt_32_multipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__F16C_C32, channels_gt_32_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(79)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_div_8_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_eq_16_fulltile) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_eq_16_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_eq_16_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_eq_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_eq_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_div_16_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_div_16_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_lt_16_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_lt_16_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_lt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_lt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_gt_16_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_gt_16_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_gt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C16, channels_gt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c16, xnn_init_f16_scaleminmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_eq_24_fulltile) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_eq_24_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_eq_24_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_eq_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_eq_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_div_24_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_div_24_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_lt_24_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_lt_24_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_lt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_lt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_gt_24_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_gt_24_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_gt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C24, channels_gt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c24, xnn_init_f16_scaleminmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_eq_32_fulltile) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_eq_32_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(32)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_eq_32_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .input_stride(37)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_eq_32_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_eq_32_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(32)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_div_32_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 64; channels < 256; channels += 32) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_div_32_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 64; channels < 256; channels += 32) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_lt_32_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_lt_32_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_lt_32_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_lt_32_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_gt_32_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_gt_32_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_gt_32_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__F16C_C32, channels_gt_32_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 33; channels < 64; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c32, xnn_init_f16_scaleminmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
