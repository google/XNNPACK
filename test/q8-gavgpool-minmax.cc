// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(8)
        .input_scale(input_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(8)
        .input_zero_point(input_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(8)
        .output_scale(output_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(8)
        .output_zero_point(output_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 128; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 128; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .input_scale(input_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .input_zero_point(input_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .output_scale(output_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .output_zero_point(output_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .input_scale(input_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .input_zero_point(input_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .output_scale(output_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .output_zero_point(output_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(8)
        .input_scale(input_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(8)
        .input_zero_point(input_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(8)
        .output_scale(output_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(8)
        .output_zero_point(output_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(8)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 128; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 128; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 128; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 128; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      for (size_t channels = 1; channels < 8; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .input_scale(input_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      for (size_t channels = 1; channels < 8; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .input_zero_point(input_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      for (size_t channels = 1; channels < 8; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .output_scale(output_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      for (size_t channels = 1; channels < 8; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .output_zero_point(output_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      for (size_t channels = 9; channels < 16; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .input_scale(input_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      for (size_t channels = 9; channels < 16; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .input_zero_point(input_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      for (size_t channels = 9; channels < 16; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .output_scale(output_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      for (size_t channels = 9; channels < 16; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .output_zero_point(output_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(8)
        .input_scale(input_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(8)
        .input_zero_point(input_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(8)
        .output_scale(output_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(8)
        .output_zero_point(output_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 128; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 128; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .input_scale(input_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .input_zero_point(input_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .output_scale(output_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .output_zero_point(output_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .input_scale(input_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .input_zero_point(input_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .output_scale(output_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .rows(7)
          .channels(channels)
          .output_zero_point(output_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(8)
        .input_scale(input_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(8)
        .input_zero_point(input_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(8)
        .output_scale(output_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(8)
        .output_zero_point(output_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(8)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 128; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 128; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 128; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 128; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      for (size_t channels = 1; channels < 8; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .input_scale(input_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      for (size_t channels = 1; channels < 8; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .input_zero_point(input_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      for (size_t channels = 1; channels < 8; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .output_scale(output_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      for (size_t channels = 1; channels < 8; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .output_zero_point(output_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      for (size_t channels = 9; channels < 16; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .input_scale(input_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      for (size_t channels = 9; channels < 16; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .input_zero_point(input_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      for (size_t channels = 9; channels < 16; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .output_scale(output_scale)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      for (size_t channels = 9; channels < 16; channels++) {
        GAvgPoolMicrokernelTester()
          .rows(14)
          .channels(channels)
          .output_zero_point(output_zero_point)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_zero_point(128)
        .output_zero_point(128)
        .input_scale(1.0f)
        .output_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .input_stride(11)
    .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_input_scale) {
  for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .input_scale(input_scale)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .input_zero_point(input_zero_point)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_output_scale) {
  for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .output_scale(output_scale)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_output_zero_point) {
  for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .output_zero_point(output_zero_point)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .input_zero_point(128)
    .output_zero_point(128)
    .input_scale(1.0f)
    .output_scale(1.0f)
    .qmax(128)
    .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .input_zero_point(128)
    .output_zero_point(128)
    .input_scale(1.0f)
    .output_scale(1.0f)
    .qmin(128)
    .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_subtile) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_input_scale) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_scale(input_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_input_zero_point) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .input_zero_point(input_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_output_scale) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .output_scale(output_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_output_zero_point) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .output_zero_point(output_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .channel_tile(8)
    .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .channel_tile(8)
    .input_stride(11)
    .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_input_scale) {
  for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .input_scale(input_scale)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .input_zero_point(input_zero_point)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_output_scale) {
  for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .output_scale(output_scale)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_output_zero_point) {
  for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .output_zero_point(output_zero_point)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .channel_tile(8)
    .input_zero_point(128)
    .output_zero_point(128)
    .input_scale(1.0f)
    .output_scale(1.0f)
    .qmax(128)
    .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .channel_tile(8)
    .input_zero_point(128)
    .output_zero_point(128)
    .input_scale(1.0f)
    .output_scale(1.0f)
    .qmin(128)
    .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(7 + rows)
      .channels(1)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_subtile_with_input_stride) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(7 + rows)
      .channels(1)
      .input_stride(11)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_input_scale) {
  for (float input_scale = 0.01f; input_scale < 100.0f; input_scale *= 3.14159265f) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_scale(input_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .input_zero_point(input_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_output_scale) {
  for (float output_scale = 0.01f; output_scale < 100.0f; output_scale *= 3.14159265f) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .output_scale(output_scale)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_output_zero_point) {
  for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .output_zero_point(output_zero_point)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .input_zero_point(128)
      .output_zero_point(128)
      .input_scale(1.0f)
      .output_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_subtile) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .Test(xnn_q8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}
