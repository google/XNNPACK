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


#if XNN_ARCH_ARM64
  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
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
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
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
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7X__NEONFP16ARITH_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
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
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
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
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
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
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
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
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
      }
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
    }
  }

  TEST(F16_GAVGPOOL_MINMAX_7P7X__NEONFP16ARITH_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
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
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
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
          .Test(xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c8);
      }
    }
  }
#endif  // XNN_ARCH_ARM64
