// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-gavgpool-minmax.yaml
//   Generator: tools/generate-gavgpool-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C8_ACC2, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c8_acc2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_eq_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_eq_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_lt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_lt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_gt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_gt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C16_ACC2, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c16_acc2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_eq_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_eq_24_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_eq_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_eq_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_eq_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_eq_24_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_eq_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_eq_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_div_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_div_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_div_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_div_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_lt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_lt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_lt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_lt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_lt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_lt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_gt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_gt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_gt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_gt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_gt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE2_C24_ACC2, channels_gt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C8_ACC2, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_eq_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_eq_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_lt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_lt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_gt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_gt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C16_ACC2, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c16_acc2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_eq_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_eq_24_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_eq_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_eq_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_eq_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_eq_24_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_eq_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_eq_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_div_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_div_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_div_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_div_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_lt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_lt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_lt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_lt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_lt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_lt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_gt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_gt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_gt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_gt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_gt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSSE3_C24_ACC2, channels_gt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c24_acc2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_eq_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_eq_8_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_eq_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_eq_8_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_eq_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_eq_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .input_stride(11)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_div_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_div_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_div_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_div_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_lt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_lt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_lt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_lt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(11)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_gt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_gt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_gt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C8_ACC2, channels_gt_8_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c8_acc2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_eq_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_eq_16_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_eq_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_eq_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_eq_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_eq_16_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_eq_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_eq_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .input_stride(19)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_div_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_div_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_div_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_div_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(263)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_lt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_lt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_lt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_lt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_lt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_lt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_gt_16_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_gt_16_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_gt_16_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_gt_16_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_gt_16_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C16_ACC2, channels_gt_16_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(47)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c16_acc2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_eq_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_eq_24_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_eq_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_eq_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_eq_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_eq_24_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_eq_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_eq_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .input_stride(29)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_div_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_div_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_div_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_div_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(389)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_lt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_lt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_lt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_lt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_lt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_lt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(29)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_gt_24_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_gt_24_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_gt_24_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_gt_24_2pass_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_gt_24_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7P7X__SSE41_C24_ACC2, channels_gt_24_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(61)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse41_c24_acc2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_eq_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_div_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_lt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_gt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C8_ACC2, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c8_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_eq_16_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_eq_16_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_eq_16_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_eq_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_eq_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_div_16_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_div_16_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_lt_16_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_lt_16_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_lt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_lt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_gt_16_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_gt_16_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_gt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C16_ACC2, channels_gt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c16_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_eq_24_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_eq_24_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_eq_24_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_eq_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_eq_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_div_24_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_div_24_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_lt_24_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_lt_24_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_lt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_lt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_gt_24_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_gt_24_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_gt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE2_C24_ACC2, channels_gt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse2_c24_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_eq_8_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_eq_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_div_8_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_div_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_lt_8_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_lt_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_gt_8_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_gt_8_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C8_ACC2, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_eq_16_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_eq_16_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_eq_16_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_eq_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_eq_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_div_16_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_div_16_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_lt_16_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_lt_16_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_lt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_lt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_gt_16_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_gt_16_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_gt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C16_ACC2, channels_gt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c16_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_eq_24_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_eq_24_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_eq_24_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_eq_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_eq_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_div_24_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_div_24_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_lt_24_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_lt_24_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_lt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_lt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_gt_24_fulltile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_gt_24_subtile) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_gt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSSE3_C24_ACC2, channels_gt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c24_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_eq_8_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(8)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_eq_8_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .input_stride(11)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(8)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_div_8_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_div_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_lt_8_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_gt_8_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C8_ACC2, channels_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_eq_16_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_eq_16_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(16)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_eq_16_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .input_stride(19)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_eq_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_eq_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(16)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_div_16_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_div_16_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 32; channels < 128; channels += 16) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_lt_16_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_lt_16_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_lt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_lt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 16; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_gt_16_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_gt_16_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_gt_16_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C16_ACC2, channels_gt_16_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 17; channels < 32; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c16_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_eq_24_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_eq_24_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(24)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_eq_24_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .input_stride(29)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_eq_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmax(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_eq_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(24)
      .qmin(128)
      .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_div_24_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_div_24_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 48; channels < 192; channels += 24) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_lt_24_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_lt_24_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_lt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_lt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 24; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_gt_24_fulltile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_gt_24_subtile) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
      }
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_gt_24_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
    }
  }

  TEST(QS8_GAVGPOOL_MINMAX_7X__SSE41_C24_ACC2, channels_gt_24_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 25; channels < 48; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c24_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
