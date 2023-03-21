// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-dwconv-minmax-unipass-rndnu.yaml
//   Generator: tools/generate-dwconv-unipass-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MLA8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P8C__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MLA8_LD128, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL8_LD128, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P16C__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, c_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(32)
      .kernel_tile(9)
      .channels(32)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, c_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, c_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, c_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(32)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(163)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_9P32C__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .channel_tile(32)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(25)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MLA8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(25)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(25)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P8C__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MLA8_LD128, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL8_LD128, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(16)
      .kernel_tile(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .channel_tile(16)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P16C__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .channel_tile(16)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, c_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .channel_tile(32)
      .kernel_tile(25)
      .channels(32)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, c_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, c_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, c_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(32)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(163)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .channel_tile(32)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_25P32C__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .channel_tile(32)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .channel_tile(1)
    .kernel_tile(9)
    .channels(1)
    .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .width(5)
      .output_stride(7)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P1C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p1c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .channel_tile(2)
    .kernel_tile(9)
    .channels(2)
    .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, c_div_2_with_qmin) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, c_div_2_with_qmax) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, c_gt_2_with_qmin) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, c_gt_2_with_qmax) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .width(5)
      .output_stride(13)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P2C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p2c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, c_eq_4) {
  DWConvMicrokernelTester()
    .channel_tile(4)
    .kernel_tile(9)
    .channels(4)
    .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, c_div_4) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, c_div_4_with_qmin) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, c_div_4_with_qmax) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, c_lt_4) {
  for (uint32_t channels = 1; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, c_gt_4) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, c_gt_4_with_qmin) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, c_gt_4_with_qmax) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .width(5)
      .output_stride(23)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, input_offset) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(channels)
      .input_offset(112)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_DWCONV_MINMAX_RNDNU_9P4C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(112)
        .zero_index(mz)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_9p4c__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}