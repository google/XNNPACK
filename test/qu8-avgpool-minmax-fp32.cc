// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-avgpool-minmax-fp32.yaml
//   Generator: tools/generate-avgpool-test.py


#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pavgpool.h"
#include "avgpool-microkernel-tester.h"
#include "next_prime.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .input_offset(11)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_offset(11)
        .zero_index_mod2(zero_index_mod2)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .output_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .output_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .qmin(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .qmax(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(41)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(41)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(67)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(11)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(17)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_multipass_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_multipass_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_multipass_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_multipass_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_multipass_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_eq_8_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_multipass_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(67)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_multipass_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_multipass_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_multipass_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_multipass_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_div_8_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(channel_tile)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_multipass_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(channel_tile)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_multipass_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_multipass_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_multipass_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_multipass_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_lt_8_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_multipass_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(17)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_multipass_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_multipass_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_multipass_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_multipass_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, channels_gt_8_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(43)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .input_offset(43)
              .zero_index_mod2(zero_index_mod2)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .input_scale(scale)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .output_scale(scale)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(128)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(128)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(43)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__NEON_C8, few_output_pixels_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(43)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .input_offset(11)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_offset(11)
        .zero_index_mod2(zero_index_mod2)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .output_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .output_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .qmin(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .qmax(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(41)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(41)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(67)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(11)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(17)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_multipass) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_multipass_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_multipass_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_multipass_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_multipass_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_multipass_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channel_tile)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_eq_8_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_multipass_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(67)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_multipass_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_multipass_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_multipass_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_multipass_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_div_8_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_multipass) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(channel_tile)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_multipass_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(channel_tile)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_multipass_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_multipass_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_multipass_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_multipass_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_lt_8_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_multipass_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(17)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_multipass_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_multipass_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_multipass_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_multipass_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, channels_gt_8_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(43)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .input_offset(43)
              .zero_index_mod2(zero_index_mod2)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .input_scale(scale)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .output_scale(scale)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(128)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(128)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(43)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SSE2_C8, few_output_pixels_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(43)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_fulltile) {
  const size_t channel_tile = 1;
  AvgPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(channel_tile)
    .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_fulltile_with_input_offset) {
  const size_t channel_tile = 1;
  AvgPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(channel_tile)
    .input_offset(3)
    .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_fulltile_with_zero) {
  const size_t channel_tile = 1;
  for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .input_offset(3)
      .zero_index_mod2(zero_index_mod2)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_fulltile_with_input_scale) {
  const size_t channel_tile = 1;
  for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .input_scale(scale)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_fulltile_with_input_zero_point) {
  const size_t channel_tile = 1;
  for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .input_zero_point(zero_point)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_fulltile_with_output_scale) {
  const size_t channel_tile = 1;
  for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .output_scale(scale)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_fulltile_with_output_zero_point) {
  const size_t channel_tile = 1;
  for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .output_zero_point(zero_point)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_fulltile_with_qmin) {
  const size_t channel_tile = 1;
  AvgPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(channel_tile)
    .qmin(128)
    .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_fulltile_with_qmax) {
  const size_t channel_tile = 1;
  AvgPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(channel_tile)
    .qmax(128)
    .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_subtile) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_subtile_with_input_offset) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .input_offset(3)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_twopass_subtile_with_zero) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_offset(3)
        .zero_index_mod2(zero_index_mod2)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_fulltile_with_input_offset) {
  for (size_t channels = 2; channels < 10; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .input_offset(3)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_fulltile_with_zero) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .zero_index_mod2(zero_index_mod2)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_fulltile_with_input_scale) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_fulltile_with_input_zero_point) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_fulltile_with_output_scale) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .output_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_fulltile_with_output_zero_point) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .output_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_subtile) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_subtile_with_input_offset) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_twopass_subtile_with_zero) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(3)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_multipass) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_with_input_offset) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .input_offset(3)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_with_zero) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_offset(3)
        .zero_index_mod2(zero_index_mod2)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_with_input_scale) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_with_input_zero_point) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .input_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_with_output_scale) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .output_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_with_output_zero_point) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channel_tile)
        .output_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_with_qmin) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .qmin(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_eq_1_multipass_with_qmax) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(channel_tile)
      .qmax(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_multipass) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 2; channels < 10; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_with_input_offset) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 2; channels < 10; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_with_zero) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(3)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_with_input_scale) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_with_input_zero_point) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_with_output_scale) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_with_output_zero_point) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_with_qmin) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 2; channels < 10; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, channels_gt_1_multipass_with_qmax) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements = xnnpack::NextPrime(pooling_elements)) {
    for (size_t channels = 2; channels < 10; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_input_offset) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(7)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_zero) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(7)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_input_scale) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_input_zero_point) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_output_scale) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_output_zero_point) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_qmin) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_qmax) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_output_stride) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_stride(7)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9P8X__SCALAR_IMAGIC_C1, few_output_pixels_with_step) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (size_t step = 2; step <= pooling_elements; step++) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .step(step)
            .channels(channels)
            .output_stride(7)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .input_offset(11)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .input_offset(11)
        .zero_index_mod2(zero_index_mod2)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .input_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .input_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .output_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .output_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .qmin(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .qmax(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(channel_tile)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(channel_tile)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_eq_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channel_tile)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(67)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(67)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_div_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(67)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_lt_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(11)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(17)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_fulltile_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_fulltile_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, channels_gt_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(17)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .input_offset(43)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_zero) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 0)
              .channels(channels)
              .input_offset(43)
              .zero_index_mod2(zero_index_mod2)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .channels(channels)
              .input_scale(scale)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .channels(channels)
              .output_scale(scale)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .qmin(128)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .qmax(128)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .output_stride(43)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__NEON_C8, few_output_pixels_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 0)
              .step(step)
              .channels(channels)
              .output_stride(43)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__neon_c8, xnn_init_qu8_avgpool_minmax_fp32_neon_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .input_offset(11)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .input_offset(11)
        .zero_index_mod2(zero_index_mod2)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .input_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .input_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .output_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channel_tile)
        .output_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .qmin(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .qmax(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(channel_tile)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(channel_tile)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_eq_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channel_tile)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(67)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(67)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_div_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(67)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_lt_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    const size_t channel_tile = 8;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(11)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(17)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_fulltile_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_fulltile_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_fulltile_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_scale(scale)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_fulltile_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .output_zero_point(zero_point)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, channels_gt_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(17)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .input_offset(43)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_zero) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 0)
              .channels(channels)
              .input_offset(43)
              .zero_index_mod2(zero_index_mod2)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .channels(channels)
              .input_scale(scale)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .channels(channels)
              .input_zero_point(zero_point)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .channels(channels)
              .output_scale(scale)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .channels(channels)
              .output_zero_point(zero_point)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .qmin(128)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .qmax(128)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .output_stride(43)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }

  TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SSE2_C8, few_output_pixels_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 0)
              .step(step)
              .channels(channels)
              .output_stride(43)
              .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__sse2_c8, xnn_init_qu8_avgpool_minmax_fp32_sse2_params, xnn_qu8_requantize_fp32);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_fulltile) {
  const size_t channel_tile = 1;
  AvgPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9)
    .channels(channel_tile)
    .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_fulltile_with_input_offset) {
  const size_t channel_tile = 1;
  AvgPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9)
    .channels(channel_tile)
    .input_offset(3)
    .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_fulltile_with_zero) {
  const size_t channel_tile = 1;
  for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .input_offset(3)
      .zero_index_mod2(zero_index_mod2)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_fulltile_with_input_scale) {
  const size_t channel_tile = 1;
  for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .input_scale(scale)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_fulltile_with_input_zero_point) {
  const size_t channel_tile = 1;
  for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .input_zero_point(zero_point)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_fulltile_with_output_scale) {
  const size_t channel_tile = 1;
  for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .output_scale(scale)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_fulltile_with_output_zero_point) {
  const size_t channel_tile = 1;
  for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channel_tile)
      .output_zero_point(zero_point)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_fulltile_with_qmin) {
  const size_t channel_tile = 1;
  AvgPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9)
    .channels(channel_tile)
    .qmin(128)
    .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_fulltile_with_qmax) {
  const size_t channel_tile = 1;
  AvgPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9)
    .channels(channel_tile)
    .qmax(128)
    .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_subtile) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9)
      .channels(channel_tile)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_subtile_with_input_offset) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9)
      .channels(channel_tile)
      .input_offset(3)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_eq_1_unipass_subtile_with_zero) {
  const size_t channel_tile = 1;
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(channel_tile)
        .input_offset(3)
        .zero_index_mod2(zero_index_mod2)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channels)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_fulltile_with_input_offset) {
  for (size_t channels = 2; channels < 10; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channels)
      .input_offset(3)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_fulltile_with_zero) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(3)
        .zero_index_mod2(zero_index_mod2)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_fulltile_with_input_scale) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_fulltile_with_input_zero_point) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_fulltile_with_output_scale) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .output_scale(scale)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_fulltile_with_output_zero_point) {
  for (size_t channels = 2; channels < 10; channels++) {
    for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .output_zero_point(zero_point)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_subtile) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_subtile_with_input_offset) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, channels_gt_1_unipass_subtile_with_zero) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(3)
          .zero_index_mod2(zero_index_mod2)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 0)
          .channels(channels)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_input_offset) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 0)
          .channels(channels)
          .input_offset(7)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_zero) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (size_t zero_index_mod2 = 0; zero_index_mod2 < 2; zero_index_mod2++) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .input_offset(7)
            .zero_index_mod2(zero_index_mod2)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_input_scale) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_input_zero_point) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_output_scale) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (float scale = 0.01f; scale < 100.0f; scale *= 3.14159265f) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .output_scale(scale)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_output_zero_point) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (int32_t zero_point = 0; zero_point <= 255; zero_point += 51) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .output_zero_point(zero_point)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_qmin) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 0)
          .channels(channels)
          .qmin(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_qmax) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 0)
          .channels(channels)
          .qmax(128)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_output_stride) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        AvgPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 0)
          .channels(channels)
          .output_stride(7)
          .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
      }
    }
  }
}

TEST(QU8_AVGPOOL_MINMAX_FP32_9X__SCALAR_IMAGIC_C1, few_output_pixels_with_step) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (size_t step = 2; step <= pooling_elements; step++) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .step(step)
            .channels(channels)
            .output_stride(7)
            .Test(xnn_qu8_avgpool_minmax_fp32_ukernel_9x__scalar_imagic_c1, xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params, xnn_qu8_requantize_fp32);
        }
      }
    }
  }
}