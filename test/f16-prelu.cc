// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-prelu.yaml
//   Generator: tools/generate-prelu-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/prelu.h"
#include "prelu-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_PRELU__NEONFP16ARITH_2X8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X8, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X8, rows_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X8, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X8, input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X8, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x8);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_PRELU__NEONFP16ARITH_2X16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X16, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X16, rows_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X16, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X16, input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X16, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
      }
    }
  }

  TEST(F16_PRELU__NEONFP16ARITH_2X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__neonfp16arith_2x16);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_PRELU__F16C_2X8, channels_eq_8) {
    TEST_REQUIRES_X86_F16C;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f16_prelu_ukernel__f16c_2x8);
  }

  TEST(F16_PRELU__F16C_2X8, channels_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__f16c_2x8);
    }
  }

  TEST(F16_PRELU__F16C_2X8, channels_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__f16c_2x8);
    }
  }

  TEST(F16_PRELU__F16C_2X8, channels_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__f16c_2x8);
    }
  }

  TEST(F16_PRELU__F16C_2X8, rows_lt_2) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__f16c_2x8);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X8, rows_div_2) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__f16c_2x8);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X8, rows_gt_2) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__f16c_2x8);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X8, input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__f16c_2x8);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X8, output_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__f16c_2x8);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X8, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__f16c_2x8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_PRELU__F16C_2X16, channels_eq_16) {
    TEST_REQUIRES_X86_F16C;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_f16_prelu_ukernel__f16c_2x16);
  }

  TEST(F16_PRELU__F16C_2X16, channels_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__f16c_2x16);
    }
  }

  TEST(F16_PRELU__F16C_2X16, channels_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__f16c_2x16);
    }
  }

  TEST(F16_PRELU__F16C_2X16, channels_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f16_prelu_ukernel__f16c_2x16);
    }
  }

  TEST(F16_PRELU__F16C_2X16, rows_lt_2) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__f16c_2x16);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X16, rows_div_2) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__f16c_2x16);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X16, rows_gt_2) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f16_prelu_ukernel__f16c_2x16);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X16, input_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__f16c_2x16);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X16, output_stride) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__f16c_2x16);
      }
    }
  }

  TEST(F16_PRELU__F16C_2X16, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f16_prelu_ukernel__f16c_2x16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
