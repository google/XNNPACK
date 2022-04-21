// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-maxpool-minmax.yaml
//   Generator: tools/generate-maxpool-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/maxpool.h>
#include "maxpool-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(23)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_eq_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_div_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(4)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_lt_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, channels_gt_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, few_output_pixels) {
    TEST_REQUIRES_X86_SSE;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, few_output_pixels_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(-16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, few_output_pixels_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__SSE_C4, few_output_pixels_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, xnn_init_f32_minmax_sse_params);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(23)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_eq_4_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_div_4_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(4)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_lt_4_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, channels_gt_4_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, few_output_pixels) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, few_output_pixels_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(-16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, few_output_pixels_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__NEON_C4, few_output_pixels_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, xnn_init_f32_minmax_scalar_params);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_unipass_fulltile) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_unipass_fulltile_with_input_offset) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_unipass_fulltile_with_qmin) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_unipass_fulltile_with_qmax) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_unipass_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_unipass_fulltile_with_input_offset) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_unipass_fulltile_with_qmin) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_unipass_fulltile_with_qmax) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_unipass_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_unipass_fulltile_with_input_offset) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_unipass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_unipass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_unipass_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_unipass_fulltile_with_input_offset) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_unipass_fulltile_with_qmin) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_unipass_fulltile_with_qmax) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_twopass_fulltile) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_twopass_fulltile_with_input_offset) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_twopass_fulltile_with_qmin) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_twopass_fulltile_with_qmax) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_twopass_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_twopass_fulltile_with_input_offset) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(23)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_twopass_fulltile_with_qmin) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_twopass_fulltile_with_qmax) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_twopass_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_twopass_fulltile_with_input_offset) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_twopass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_twopass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_twopass_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_twopass_fulltile_with_input_offset) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_twopass_fulltile_with_qmin) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_twopass_fulltile_with_qmax) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_eq_4_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_div_4_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(4)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_lt_4_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, channels_gt_4_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, few_output_pixels) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, few_output_pixels_with_input_offset) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, few_output_pixels_with_qmin) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(-16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, few_output_pixels_with_qmax) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, few_output_pixels_with_output_stride) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_ARM_C4, few_output_pixels_with_step) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, xnn_init_f32_minmax_wasmsimd_params);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_unipass_fulltile) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_unipass_fulltile_with_input_offset) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_unipass_fulltile_with_qmin) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_unipass_fulltile_with_qmax) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_unipass_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_unipass_fulltile_with_input_offset) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_unipass_fulltile_with_qmin) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_unipass_fulltile_with_qmax) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_unipass_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_unipass_fulltile_with_input_offset) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_unipass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_unipass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_unipass_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_unipass_fulltile_with_input_offset) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_unipass_fulltile_with_qmin) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_unipass_fulltile_with_qmax) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_twopass_fulltile) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_twopass_fulltile_with_input_offset) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_twopass_fulltile_with_qmin) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_twopass_fulltile_with_qmax) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_twopass_fulltile) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_twopass_fulltile_with_input_offset) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(23)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_twopass_fulltile_with_qmin) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_twopass_fulltile_with_qmax) {
    for (size_t channels = 8; channels < 32; channels += 4) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_twopass_fulltile) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_twopass_fulltile_with_input_offset) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_twopass_fulltile_with_qmin) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_twopass_fulltile_with_qmax) {
    for (size_t channels = 1; channels < 4; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_twopass_fulltile) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_twopass_fulltile_with_input_offset) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_twopass_fulltile_with_qmin) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_twopass_fulltile_with_qmax) {
    for (size_t channels = 5; channels < 8; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_eq_4_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_div_4_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(4)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_lt_4_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, channels_gt_4_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, few_output_pixels) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, few_output_pixels_with_input_offset) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, few_output_pixels_with_qmin) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(-16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, few_output_pixels_with_qmax) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, few_output_pixels_with_output_stride) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASMSIMD_X86_C4, few_output_pixels_with_step) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, xnn_init_f32_minmax_wasmsimd_params);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_unipass_fulltile) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_unipass_fulltile_with_input_offset) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_unipass_fulltile_with_qmin) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(1)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_unipass_fulltile_with_qmax) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(1)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(1)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(1)
        .input_offset(3)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_unipass_fulltile) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_unipass_fulltile_with_input_offset) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_unipass_fulltile_with_qmin) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_unipass_fulltile_with_qmax) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_unipass_subtile) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 2; channels < 10; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_unipass_subtile_with_input_offset) {
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 2; channels < 10; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(3)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_twopass_fulltile) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_twopass_fulltile_with_input_offset) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_twopass_fulltile_with_qmin) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(1)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_twopass_fulltile_with_qmax) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(1)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(1)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(1)
        .input_offset(3)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_twopass_fulltile) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_twopass_fulltile_with_input_offset) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_twopass_fulltile_with_qmin) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_twopass_fulltile_with_qmax) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_twopass_subtile) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 2; channels < 10; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_twopass_subtile_with_input_offset) {
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 2; channels < 10; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(3)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(1)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(1)
        .input_offset(3)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(1)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_eq_1_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(1)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_multipass) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 2; channels < 10; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_multipass_with_input_offset) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 2; channels < 10; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(3)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_multipass_with_qmin) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 2; channels < 10; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, channels_gt_1_multipass_with_qmax) {
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 2; channels < 10; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, few_output_pixels) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 5; channels += 1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, few_output_pixels_with_input_offset) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 5; channels += 1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(7)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, few_output_pixels_with_qmin) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 5; channels += 1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(-16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, few_output_pixels_with_qmax) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 5; channels += 1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(16384)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, few_output_pixels_with_output_stride) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 5; channels += 1) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(7)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_MAXPOOL_MINMAX_9P8X__WASM_C1, few_output_pixels_with_step) {
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 5; channels += 1) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(7)
              .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, xnn_init_f32_minmax_scalar_params);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_fulltile) {
  MaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9, 8)
    .channels(1)
    .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_input_offset) {
  MaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9, 8)
    .channels(1)
    .input_offset(3)
    .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_qmin) {
  MaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9, 8)
    .channels(1)
    .qmin(-16384)
    .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_qmax) {
  MaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9, 8)
    .channels(1)
    .qmax(16384)
    .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_subtile) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_subtile_with_input_offset) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(channels)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_input_offset) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(channels)
      .input_offset(3)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_subtile) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_subtile_with_input_offset) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile) {
  MaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile_with_input_offset) {
  MaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .input_offset(3)
    .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile_with_qmin) {
  MaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .qmin(-16384)
    .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile_with_qmax) {
  MaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .qmax(16384)
    .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_subtile) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_subtile_with_input_offset) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile_with_input_offset) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .input_offset(3)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_subtile) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_subtile_with_input_offset) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_multipass) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_multipass_with_input_offset) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_multipass_with_qmin) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .qmin(-16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_multipass_with_qmax) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .qmax(16384)
      .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_multipass) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_multipass_with_input_offset) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_multipass_with_qmin) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(-16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_multipass_with_qmax) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(16384)
        .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_input_offset) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(7)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_qmin) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(-16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_qmax) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(16384)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_output_stride) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_stride(7)
          .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_step) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (size_t step = 2; step <= pooling_elements; step++) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .step(step)
            .channels(channels)
            .output_stride(7)
            .Test(xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }
}