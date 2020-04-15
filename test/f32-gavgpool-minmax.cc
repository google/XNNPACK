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
  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .input_stride(11)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_div_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 4; channels < 64; channels += 12) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_lt_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_lt_4_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_lt_4_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_gt_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_gt_4_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_gt_4_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .input_stride(11)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(4)
        .input_stride(11)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_div_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 4; channels < 64; channels += 12) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_div_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_div_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_div_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_eq_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_eq_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_eq_4_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .input_stride(11)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_eq_4_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_eq_4_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_div_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 4; channels < 64; channels += 12) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_lt_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_lt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_lt_4_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_lt_4_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_gt_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_gt_4_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE2_C4, channels_gt_4_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_eq_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .input_stride(11)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_eq_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_eq_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_eq_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_eq_4_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(4)
        .input_stride(11)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_eq_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_div_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 4; channels < 64; channels += 12) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_div_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_div_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_div_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_lt_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_lt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_lt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_lt_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_lt_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_gt_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_gt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_gt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_gt_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_gt_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE2_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_eq_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_eq_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_eq_4_fulltile_with_input_stride) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .input_stride(11)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_eq_4_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_eq_4_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_div_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 4; channels < 64; channels += 12) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_div_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_lt_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_lt_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_lt_4_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_lt_4_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_gt_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_gt_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_gt_4_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__PSIMD_C4, channels_gt_4_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_eq_4_2pass_fulltile) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .input_stride(11)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_eq_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_eq_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_eq_4_2pass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_eq_4_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(4)
        .input_stride(11)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_eq_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_div_4_2pass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 4; channels < 64; channels += 12) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_div_4_2pass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_div_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_div_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 4; channels < 64; channels += 12) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(131)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_lt_4_2pass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_lt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_lt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_lt_4_2pass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_lt_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_gt_4_2pass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_gt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_gt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_gt_4_2pass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_gt_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__PSIMD_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__psimd_c4, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if XNN_ARCH_WASM
  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_subtile) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(1)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .input_stride(11)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_gt_1_fulltile) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_gt_1_subtile) {
    for (size_t channels = 2; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_gt_1_fulltile_with_qmax) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_gt_1_fulltile_with_qmin) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .input_stride(11)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_subtile) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(1)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_subtile_with_input_stride) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(1)
          .input_stride(11)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_multipass_fulltile) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(1)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_multipass_fulltile_with_input_stride) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(1)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_2pass_fulltile) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_2pass_fulltile_with_qmax) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
          .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_2pass_fulltile_with_qmin) {
    for (size_t channels = 2; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
          .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_2pass_subtile) {
    for (size_t channels = 2; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(7 + rows)
          .channels(channels)
              .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_multipass_fulltile) {
    for (size_t channels = 2; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
              .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_multipass_fulltile_with_input_stride) {
    for (size_t channels = 2; channels < 8; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
              .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // XNN_ARCH_WASM


TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .input_stride(11)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmax(128)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmin(128)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_subtile) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .input_stride(11)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmax(128)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmin(128)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(7 + rows)
      .channels(1)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_subtile_with_input_stride) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(7 + rows)
      .channels(1)
        .input_stride(11)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
        .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 8; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
        .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_subtile) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(7 + rows)
        .channels(channels)
            .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
            .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 8; channels++) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
            .input_stride(23)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}
