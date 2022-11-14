// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-ibilinear-chw.yaml
//   Generator: tools/generate-ibilinear-chw-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/ibilinear.h>
#include "ibilinear-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P4, pixels_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    IBilinearMicrokernelTester()
      .pixels(4)
      .channels(1)
      .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p4);
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P4, pixels_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 8; pixels < 40; pixels += 4) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p4);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P4, pixels_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 4; pixels++) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p4);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P4, pixels_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 5; pixels < 8; pixels++) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p4);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P4, channels_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels <= 20; pixels += 3) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p4);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P4, channels_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 2; channels < 3; channels++) {
      for (size_t pixels = 1; pixels <= 20; pixels += 3) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p4);
      }
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P4, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 20; pixels += 3) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(7)
          .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p4);
      }
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P4, input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 20; pixels += 3) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_stride(83)
          .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p4);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P8, pixels_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    IBilinearMicrokernelTester()
      .pixels(8)
      .channels(1)
      .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8);
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P8, pixels_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 16; pixels < 80; pixels += 8) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P8, pixels_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 8; pixels++) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P8, pixels_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 9; pixels < 16; pixels++) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P8, channels_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels <= 40; pixels += 7) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P8, channels_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 2; channels < 3; channels++) {
      for (size_t pixels = 1; pixels <= 40; pixels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8);
      }
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P8, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 40; pixels += 7) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(7)
          .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8);
      }
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P8, input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 40; pixels += 7) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_stride(163)
          .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P16, pixels_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    IBilinearMicrokernelTester()
      .pixels(16)
      .channels(1)
      .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16);
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P16, pixels_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 32; pixels < 160; pixels += 16) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P16, pixels_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 16; pixels++) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P16, pixels_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 17; pixels < 32; pixels++) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P16, channels_eq_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels <= 80; pixels += 15) {
      IBilinearMicrokernelTester()
        .pixels(pixels)
        .channels(1)
        .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16);
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P16, channels_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 2; channels < 3; channels++) {
      for (size_t pixels = 1; pixels <= 80; pixels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16);
      }
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P16, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 80; pixels += 15) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(7)
          .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16);
      }
    }
  }

  TEST(F16_IBILINEAR_CHW__NEONFP16ARITH_P16, input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 80; pixels += 15) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_stride(331)
          .TestCHW(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
