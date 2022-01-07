// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-gavgpool-minmax.yaml
//   Generator: tools/generate-gavgpool-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__NEON_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__neon_c8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__NEON_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__neon_c8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_div_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7X__SSE2_C8, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }

  TEST(QU8_GAVGPOOL_MINMAX_7P7X__SSE2_C8, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__sse2_c8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(QU8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(QU8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .input_stride(3)
    .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(QU8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmax(128)
    .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(QU8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmin(128)
    .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(QU8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .input_stride(3)
    .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmax(128)
    .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmin(128)
    .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_div_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_div_1_2pass_subtile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_div_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_div_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(19)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(QU8_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(17)
        .Test(xnn_qu8_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}
