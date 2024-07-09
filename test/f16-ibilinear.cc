// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-ibilinear.yaml
//   Generator: tools/generate-ibilinear-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/isa-checks.h"
#include "ibilinear-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_IBILINEAR__NEONFP16ARITH_C8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c8);
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c8);
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c8);
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c8);
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C8, pixels_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c8);
      }
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C8, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c8);
      }
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C8, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c8);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_IBILINEAR__NEONFP16ARITH_C16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(16)
      .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c16);
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 32; channels < 160; channels += 16) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c16);
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c16);
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c16);
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C16, pixels_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c16);
      }
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C16, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(83)
          .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c16);
      }
    }
  }

  TEST(F16_IBILINEAR__NEONFP16ARITH_C16, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(83)
          .Test(xnn_f16_ibilinear_ukernel__neonfp16arith_c16);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IBILINEAR__FMA3_C8, channels_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(8)
      .Test(xnn_f16_ibilinear_ukernel__fma3_c8);
  }

  TEST(F16_IBILINEAR__FMA3_C8, channels_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 16; channels < 80; channels += 8) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__fma3_c8);
    }
  }

  TEST(F16_IBILINEAR__FMA3_C8, channels_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels < 8; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__fma3_c8);
    }
  }

  TEST(F16_IBILINEAR__FMA3_C8, channels_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 9; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__fma3_c8);
    }
  }

  TEST(F16_IBILINEAR__FMA3_C8, pixels_gt_1) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f16_ibilinear_ukernel__fma3_c8);
      }
    }
  }

  TEST(F16_IBILINEAR__FMA3_C8, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(43)
          .Test(xnn_f16_ibilinear_ukernel__fma3_c8);
      }
    }
  }

  TEST(F16_IBILINEAR__FMA3_C8, output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(43)
          .Test(xnn_f16_ibilinear_ukernel__fma3_c8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_IBILINEAR__FMA3_C16, channels_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    IBilinearMicrokernelTester()
      .pixels(1)
      .channels(16)
      .Test(xnn_f16_ibilinear_ukernel__fma3_c16);
  }

  TEST(F16_IBILINEAR__FMA3_C16, channels_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 32; channels < 160; channels += 16) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__fma3_c16);
    }
  }

  TEST(F16_IBILINEAR__FMA3_C16, channels_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels < 16; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__fma3_c16);
    }
  }

  TEST(F16_IBILINEAR__FMA3_C16, channels_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 17; channels < 32; channels++) {
      IBilinearMicrokernelTester()
        .pixels(1)
        .channels(channels)
        .Test(xnn_f16_ibilinear_ukernel__fma3_c16);
    }
  }

  TEST(F16_IBILINEAR__FMA3_C16, pixels_gt_1) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t pixels = 2; pixels < 3; pixels++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .Test(xnn_f16_ibilinear_ukernel__fma3_c16);
      }
    }
  }

  TEST(F16_IBILINEAR__FMA3_C16, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .input_offset(83)
          .Test(xnn_f16_ibilinear_ukernel__fma3_c16);
      }
    }
  }

  TEST(F16_IBILINEAR__FMA3_C16, output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t pixels = 1; pixels < 5; pixels += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        IBilinearMicrokernelTester()
          .pixels(pixels)
          .channels(channels)
          .output_stride(83)
          .Test(xnn_f16_ibilinear_ukernel__fma3_c16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
