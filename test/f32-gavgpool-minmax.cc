// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-gavgpool-minmax.yaml
//   Generator: tools/generate-gavgpool-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .input_stride(7)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .input_stride(7)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_div_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_div_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_div_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_div_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_lt_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
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
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
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
          .input_stride(7)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__NEON_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_eq_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_eq_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_eq_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_eq_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_eq_4_2pass_subtile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .input_stride(7)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_eq_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .input_stride(7)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_div_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_div_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_div_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_div_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_lt_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_lt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_lt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_lt_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_lt_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(7)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_gt_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_gt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_gt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_gt_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_gt_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__SSE_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_eq_4_2pass_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_eq_4_2pass_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_eq_4_2pass_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_eq_4_2pass_subtile) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_eq_4_2pass_subtile_with_input_stride) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .input_stride(7)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_eq_4_multipass_fulltile) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .input_stride(7)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_div_4_2pass_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_div_4_2pass_subtile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_div_4_multipass_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_div_4_multipass_fulltile_with_input_stride) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_lt_4_2pass_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_lt_4_2pass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_lt_4_2pass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_lt_4_2pass_subtile) {
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_lt_4_multipass_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(7)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_gt_4_2pass_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_gt_4_2pass_fulltile_with_qmax) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_gt_4_2pass_fulltile_with_qmin) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_gt_4_2pass_subtile) {
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_gt_4_multipass_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_ARM_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_eq_4_2pass_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_eq_4_2pass_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_eq_4_2pass_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_eq_4_2pass_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_eq_4_2pass_subtile) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_eq_4_2pass_subtile_with_input_stride) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .input_stride(7)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_eq_4_multipass_fulltile) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_eq_4_multipass_fulltile_with_input_stride) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .input_stride(7)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_div_4_2pass_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_div_4_2pass_subtile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_div_4_multipass_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_div_4_multipass_fulltile_with_input_stride) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(67)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_lt_4_2pass_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_lt_4_2pass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_lt_4_2pass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_lt_4_2pass_subtile) {
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_lt_4_multipass_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_lt_4_multipass_fulltile_with_input_stride) {
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(7)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_gt_4_2pass_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_gt_4_2pass_fulltile_with_qmax) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_gt_4_2pass_fulltile_with_qmin) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_gt_4_2pass_subtile) {
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_gt_4_multipass_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASMSIMD_X86_C4, channels_gt_4_multipass_fulltile_with_input_stride) {
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .input_stride(3)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(1)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_subtile) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(1)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_2pass_subtile_with_input_stride) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(1)
        .input_stride(3)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_multipass_fulltile) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(1)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_eq_1_multipass_fulltile_with_input_stride) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(1)
        .input_stride(3)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_div_1_2pass_fulltile) {
    for (size_t channels = 2; channels < 8; channels += 1) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_div_1_2pass_subtile) {
    for (size_t channels = 2; channels < 8; channels += 1) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_div_1_multipass_fulltile) {
    for (size_t channels = 2; channels < 8; channels += 1) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_div_1_multipass_fulltile_with_input_stride) {
    for (size_t channels = 2; channels < 8; channels += 1) {
      for (size_t rows = 14; rows <= 35; rows += 7) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(19)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_2pass_fulltile) {
    for (size_t channels = 2; channels < 10; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_2pass_fulltile_with_qmax) {
    for (size_t channels = 2; channels < 10; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_2pass_fulltile_with_qmin) {
    for (size_t channels = 2; channels < 10; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(14)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_2pass_subtile) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (size_t rows = 8; rows < 14; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_multipass_fulltile) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7P7X__WASM_C1, channels_gt_1_multipass_fulltile_with_input_stride) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (size_t rows = 14; rows < 35; rows += 14) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(17)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .input_stride(3)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmax(128)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(14)
    .channels(1)
    .qmin(128)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_subtile) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_2pass_subtile_with_input_stride) {
  for (size_t rows = 8; rows < 14; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_multipass_fulltile) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_eq_1_multipass_fulltile_with_input_stride) {
  for (size_t rows = 14; rows <= 35; rows += 7) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .input_stride(3)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_div_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_div_1_2pass_subtile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_div_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_div_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 8; channels += 1) {
    for (size_t rows = 14; rows <= 35; rows += 7) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(19)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(14)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_2pass_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 8; rows < 14; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_multipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_GAVGPOOL_MINMAX_7P7X__SCALAR_C1, channels_gt_1_multipass_fulltile_with_input_stride) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 14; rows < 35; rows += 14) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(17)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7p7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_fulltile_with_input_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_eq_4_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_div_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_lt_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
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
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_lt_4_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_gt_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
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
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__NEON_C4, channels_gt_4_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__neon_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_eq_4_fulltile) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_eq_4_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_eq_4_fulltile_with_input_stride) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_eq_4_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_eq_4_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_div_4_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_div_4_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_lt_4_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_lt_4_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_lt_4_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_lt_4_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_gt_4_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_gt_4_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__SSE_C4, channels_gt_4_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4, xnn_init_f32_scaleminmax_sse_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_eq_4_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_eq_4_subtile) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_eq_4_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_eq_4_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_eq_4_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_div_4_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_div_4_subtile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_lt_4_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_lt_4_subtile) {
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_lt_4_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_lt_4_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_gt_4_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_gt_4_subtile) {
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_gt_4_fulltile_with_qmax) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_ARM_C4, channels_gt_4_fulltile_with_qmin) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_arm_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_eq_4_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_eq_4_subtile) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(4)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_eq_4_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .input_stride(7)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_eq_4_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_eq_4_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_div_4_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_div_4_subtile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_lt_4_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_lt_4_subtile) {
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_lt_4_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_lt_4_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 4; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_gt_4_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_gt_4_subtile) {
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_gt_4_fulltile_with_qmax) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASMSIMD_X86_C4, channels_gt_4_fulltile_with_qmin) {
    for (size_t channels = 5; channels < 8; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasmsimd_x86_c4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_fulltile) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_subtile) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(1)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_fulltile_with_input_stride) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .input_stride(3)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_fulltile_with_qmax) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_eq_1_fulltile_with_qmin) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(1)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_gt_1_fulltile) {
    for (size_t channels = 2; channels < 10; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_gt_1_subtile) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (size_t rows = 1; rows < 7; rows++) {
        GAvgPoolMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
      }
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_gt_1_fulltile_with_qmax) {
    for (size_t channels = 2; channels < 10; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_GAVGPOOL_MINMAX_7X__WASM_C1, channels_gt_1_fulltile_with_qmin) {
    for (size_t channels = 2; channels < 10; channels++) {
      GAvgPoolMicrokernelTester()
        .rows(7)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_subtile) {
  for (size_t rows = 1; rows < 7; rows++) {
    GAvgPoolMicrokernelTester()
      .rows(rows)
      .channels(1)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_input_stride) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .input_stride(3)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmax(128)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_eq_1_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .rows(7)
    .channels(1)
    .qmin(128)
    .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_subtile) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t rows = 1; rows < 7; rows++) {
      GAvgPoolMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
    }
  }
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_GAVGPOOL_MINMAX_7X__SCALAR_C1, channels_gt_1_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    GAvgPoolMicrokernelTester()
      .rows(7)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_gavgpool_minmax_ukernel_7x__scalar_c1, xnn_init_f32_scaleminmax_scalar_params);
  }
}