// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/u8-maxpool-minmax.yaml
//   Generator: tools/generate-maxpool-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/maxpool.h>
#include "maxpool-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(16)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(16)
      .input_offset(19)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(16)
      .qmin(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(16)
      .qmax(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .input_offset(19)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(131)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(131)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(16)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(16)
      .input_offset(19)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(16)
      .qmin(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(16)
      .qmax(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .input_offset(19)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(83)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(131)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .input_offset(19)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_eq_16_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(131)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_div_16_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(16)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_lt_16_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, channels_gt_16_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, few_output_pixels) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(83)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, few_output_pixels_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(192)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, few_output_pixels_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(192)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(83)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__NEON_C16, few_output_pixels_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(83)
              .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__neon_c16);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(16)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(16)
      .input_offset(19)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(16)
      .qmin(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(16)
      .qmax(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .input_offset(19)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(131)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(131)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(16)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(16)
      .input_offset(19)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(16)
      .qmin(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(16)
      .qmax(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .input_offset(19)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(83)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 32; channels < 128; channels += 16) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(131)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 16; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 17; channels < 32; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .input_offset(19)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_eq_16_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(16)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(131)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_div_16_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 32; channels < 128; channels += 16) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(16)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_lt_16_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 16; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, channels_gt_16_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 17; channels < 32; channels++) {
        MaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, few_output_pixels) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(83)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, few_output_pixels_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(192)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, few_output_pixels_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(192)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          MaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(83)
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
        }
      }
    }
  }

  TEST(U8_MAXPOOL_MINMAX_9P8X__SSE2_C16, few_output_pixels_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
        for (size_t channels = 1; channels <= 80; channels += 15) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            MaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(83)
              .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__sse2_c16);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_fulltile) {
  MaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9, 8)
    .channels(1)
    .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_input_offset) {
  MaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9, 8)
    .channels(1)
    .input_offset(3)
    .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_qmin) {
  MaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9, 8)
    .channels(1)
    .qmin(192)
    .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_qmax) {
  MaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9, 8)
    .channels(1)
    .qmax(192)
    .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_subtile) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_unipass_subtile_with_input_offset) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(channels)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_input_offset) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(channels)
      .input_offset(3)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmin(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmax(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_subtile) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_unipass_subtile_with_input_offset) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile) {
  MaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile_with_input_offset) {
  MaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .input_offset(3)
    .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile_with_qmin) {
  MaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .qmin(192)
    .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile_with_qmax) {
  MaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .qmax(192)
    .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_subtile) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_twopass_subtile_with_input_offset) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile_with_input_offset) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .input_offset(3)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmin(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    MaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmax(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_subtile) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_twopass_subtile_with_input_offset) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_multipass) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_multipass_with_input_offset) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_multipass_with_qmin) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .qmin(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_eq_1_multipass_with_qmax) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    MaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .qmax(192)
      .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_multipass) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_multipass_with_input_offset) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_multipass_with_qmin) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, channels_gt_1_multipass_with_qmax) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      MaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_input_offset) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(7)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_qmin) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_qmax) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_output_stride) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 9, 16}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        MaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_stride(7)
          .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(U8_MAXPOOL_MINMAX_9P8X__SCALAR_C1, few_output_pixels_with_step) {
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
            .Test(xnn_u8_maxpool_minmax_ukernel_9p8x__scalar_c1, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}