// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-dwconv-minmax-rndnu.yaml
//   Generator: tools/generate-dwconv-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL8_LD128, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL8_LD128, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MLA8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MLA8_LD128, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MLA8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD64, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MLA8_LD128, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X9__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X9__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, c_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(24)
      .kr(9)
      .channels(24)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, c_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, c_div_24_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, c_div_24_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, c_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 24; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, c_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, c_gt_24_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, c_gt_24_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(24)
        .width(5)
        .output_stride(127)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(9)
        .channels(channels)
        .input_offset(464)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X9__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 48; channels < 384; channels += 72) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(9)
          .channels(channels)
          .input_offset(464)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, c_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, c_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, c_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, c_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X9__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP8X25__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, c_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, c_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, c_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, c_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP16X25__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, c_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(24)
      .kr(25)
      .channels(24)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, c_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, c_div_24_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, c_div_24_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, c_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 24; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, c_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, c_gt_24_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, c_gt_24_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 25; channels < 48; channels++) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(24)
        .width(5)
        .output_stride(127)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 120; channels += 23) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 48; channels < 384; channels += 72) {
      DWConvMicrokernelTester()
        .cr(24)
        .kr(25)
        .channels(channels)
        .input_offset(464)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP24X25__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 48; channels < 384; channels += 72) {
        DWConvMicrokernelTester()
          .cr(24)
          .kr(25)
          .channels(channels)
          .input_offset(464)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, c_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(25)
      .channels(32)
      .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, c_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, c_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, c_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_DWCONV_MINMAX_RNDNU_UP32X25__NEON_MUL16, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
