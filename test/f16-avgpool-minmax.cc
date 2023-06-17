// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-avgpool-minmax.yaml
//   Generator: tools/generate-avgpool-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/avgpool.h>
#include <xnnpack/pavgpool.h>
#include "avgpool-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(8)
      .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(8)
      .input_offset(11)
      .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t zero_index = 0; zero_index < 17; zero_index++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(8)
        .input_offset(11)
        .zero_index(zero_index)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(8)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(41)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t zero_index = 0; zero_index < 17; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(41)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(67)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_twopass_fulltile_with_zero_index) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t zero_index = 0; zero_index < 17; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(11)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t zero_index = 0; zero_index < 17; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(17)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_multipass_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(8)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_eq_8_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_multipass_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(67)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_div_8_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(8)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_multipass_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(8)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_lt_8_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_multipass_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_multipass_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(17)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_multipass_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, channels_gt_8_multipass_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, few_output_pixels) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(43)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, few_output_pixels_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .input_offset(43)
              .zero_index(zero_index)
              .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
          }
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, few_output_pixels_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(128)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, few_output_pixels_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(128)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(43)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__NEONFP16ARITH_C8, few_output_pixels_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
              .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
          }
        }
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_eq_8_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(8)
      .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_eq_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(8)
      .input_offset(11)
      .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_eq_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t zero_index = 0; zero_index < 9; zero_index++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(8)
        .input_offset(11)
        .zero_index(zero_index)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_eq_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_eq_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_eq_8_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(8)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_eq_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(8)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_eq_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(8)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_div_8_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_div_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(67)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_div_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t zero_index = 0; zero_index < 9; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(67)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_div_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_div_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_div_8_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_div_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_div_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(67)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_lt_8_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_lt_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_lt_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t zero_index = 0; zero_index < 9; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_lt_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_lt_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_lt_8_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_lt_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_lt_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(11)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_gt_8_unipass_fulltile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_gt_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_gt_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t zero_index = 0; zero_index < 9; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(17)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_gt_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_gt_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_gt_8_unipass_subtile) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_gt_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, channels_gt_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(17)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, few_output_pixels) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .input_offset(43)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, few_output_pixels_with_zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 0)
              .channels(channels)
              .input_offset(43)
              .zero_index(zero_index)
              .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
          }
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, few_output_pixels_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .qmin(128)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, few_output_pixels_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .qmax(128)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .output_stride(43)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__NEONFP16ARITH_C8, few_output_pixels_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
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
              .Test(xnn_f16_avgpool_minmax_ukernel_9x__neonfp16arith_c8, xnn_init_f16_scaleminmax_fp16arith_params);
          }
        }
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_twopass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(8)
      .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(8)
      .input_offset(11)
      .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t zero_index = 0; zero_index < 17; zero_index++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(8)
        .input_offset(11)
        .zero_index(zero_index)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    AvgPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_twopass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(8)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_twopass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(41)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t zero_index = 0; zero_index < 17; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(41)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_twopass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(67)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_twopass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_twopass_fulltile_with_zero_index) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t zero_index = 0; zero_index < 17; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_twopass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(11)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_twopass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_twopass_fulltile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t zero_index = 0; zero_index < 17; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_twopass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_twopass_subtile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(17)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_multipass) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_multipass_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_multipass_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(8)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_multipass_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_eq_8_multipass_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(8)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_multipass) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_multipass_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_multipass_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(67)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_multipass_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_div_8_multipass_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_multipass) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_multipass_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(8)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_multipass_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(8)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_multipass_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_lt_8_multipass_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_multipass) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_multipass_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_multipass_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(17)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_multipass_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, channels_gt_8_multipass_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(128)
          .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, few_output_pixels) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(43)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, few_output_pixels_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .channels(channels)
              .input_offset(43)
              .zero_index(zero_index)
              .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
          }
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, few_output_pixels_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(128)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, few_output_pixels_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(128)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{10, 16, 18}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(43)
            .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9P8X__F16C_C8, few_output_pixels_with_step) {
    TEST_REQUIRES_X86_F16C;
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
              .Test(xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_eq_8_unipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(8)
      .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_eq_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(8)
      .input_offset(11)
      .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_eq_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t zero_index = 0; zero_index < 9; zero_index++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(8)
        .input_offset(11)
        .zero_index(zero_index)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_eq_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(8)
      .qmin(128)
      .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_eq_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    AvgPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(8)
      .qmax(128)
      .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_eq_8_unipass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(8)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_eq_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(8)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_eq_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(8)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_div_8_unipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_div_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(67)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_div_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      for (size_t zero_index = 0; zero_index < 9; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(67)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_div_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_div_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 64; channels += 8) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_div_8_unipass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_div_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(67)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_div_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 16; channels < 64; channels += 8) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(67)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_lt_8_unipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_lt_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_lt_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      for (size_t zero_index = 0; zero_index < 9; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_lt_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_lt_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_lt_8_unipass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_lt_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_lt_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 8; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(11)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_gt_8_unipass_fulltile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_gt_8_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(17)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_gt_8_unipass_fulltile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      for (size_t zero_index = 0; zero_index < 9; zero_index++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(9)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(17)
          .zero_index(zero_index)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_gt_8_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_gt_8_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      AvgPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_gt_8_unipass_subtile) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_gt_8_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        AvgPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(17)
          .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, channels_gt_8_unipass_subtile_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 9; channels < 16; channels++) {
        for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
          AvgPoolMicrokernelTester()
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(17)
            .zero_index(zero_index)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, few_output_pixels) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .input_offset(43)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, few_output_pixels_with_zero) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          for (size_t zero_index = 0; zero_index < pooling_elements; zero_index++) {
            AvgPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 0)
              .channels(channels)
              .input_offset(43)
              .zero_index(zero_index)
              .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
          }
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, few_output_pixels_with_qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .qmin(128)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, few_output_pixels_with_qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .qmax(128)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, 8, 9}}) {
        for (size_t channels = 1; channels <= 40; channels += 7) {
          AvgPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 0)
            .channels(channels)
            .output_stride(43)
            .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
        }
      }
    }
  }

  TEST(F16_AVGPOOL_MINMAX_9X__F16C_C8, few_output_pixels_with_step) {
    TEST_REQUIRES_X86_F16C;
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
              .Test(xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8, xnn_init_f16_scaleminmax_avx_params);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
