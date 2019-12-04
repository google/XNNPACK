// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-prelu.yaml
//   Generator: tools/generate-prelu-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/prelu.h>
#include "prelu-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_2X4, channels_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__neon_2x4);
  }

  TEST(F32_PRELU__NEON_2X4, channels_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x4);
    }
  }

  TEST(F32_PRELU__NEON_2X4, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x4);
    }
  }

  TEST(F32_PRELU__NEON_2X4, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x4);
    }
  }

  TEST(F32_PRELU__NEON_2X4, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, rows_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_2X8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__neon_2x8);
  }

  TEST(F32_PRELU__NEON_2X8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x8);
    }
  }

  TEST(F32_PRELU__NEON_2X8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x8);
    }
  }

  TEST(F32_PRELU__NEON_2X8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x8);
    }
  }

  TEST(F32_PRELU__NEON_2X8, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, rows_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE2_2X4, channels_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__sse2_2x4);
  }

  TEST(F32_PRELU__SSE2_2X4, channels_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x4);
    }
  }

  TEST(F32_PRELU__SSE2_2X4, channels_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x4);
    }
  }

  TEST(F32_PRELU__SSE2_2X4, channels_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x4);
    }
  }

  TEST(F32_PRELU__SSE2_2X4, rows_lt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, rows_div_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, rows_gt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE2_2X8, channels_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__sse2_2x8);
  }

  TEST(F32_PRELU__SSE2_2X8, channels_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x8);
    }
  }

  TEST(F32_PRELU__SSE2_2X8, channels_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x8);
    }
  }

  TEST(F32_PRELU__SSE2_2X8, channels_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x8);
    }
  }

  TEST(F32_PRELU__SSE2_2X8, rows_lt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, rows_div_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, rows_gt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE41_2X4, channels_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__sse41_2x4);
  }

  TEST(F32_PRELU__SSE41_2X4, channels_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x4);
    }
  }

  TEST(F32_PRELU__SSE41_2X4, channels_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x4);
    }
  }

  TEST(F32_PRELU__SSE41_2X4, channels_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x4);
    }
  }

  TEST(F32_PRELU__SSE41_2X4, rows_lt_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, rows_div_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, rows_gt_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, output_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE41_2X8, channels_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__sse41_2x8);
  }

  TEST(F32_PRELU__SSE41_2X8, channels_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x8);
    }
  }

  TEST(F32_PRELU__SSE41_2X8, channels_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x8);
    }
  }

  TEST(F32_PRELU__SSE41_2X8, channels_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x8);
    }
  }

  TEST(F32_PRELU__SSE41_2X8, rows_lt_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, rows_div_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, rows_gt_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, output_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM
  TEST(F32_PRELU__PSIMD_2X4, channels_eq_4) {
    TEST_REQUIRES_PSIMD;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PRELU__PSIMD_2X4, channels_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, channels_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, channels_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, rows_lt_2) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, rows_div_2) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, rows_gt_2) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, input_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X4, qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM
  TEST(F32_PRELU__PSIMD_2X8, channels_eq_8) {
    TEST_REQUIRES_PSIMD;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PRELU__PSIMD_2X8, channels_div_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, channels_lt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, channels_gt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, rows_lt_2) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, rows_div_2) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, rows_gt_2) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, input_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__PSIMD_2X8, qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__psimd_2x8, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_PRELU__WASM_2X1, channels_eq_1) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(1)
      .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PRELU__WASM_2X1, channels_gt_1) {
    for (size_t channels = 2; channels < 10; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__WASM_2X1, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(7)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(7)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_PRELU__WASM_2X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_PRELU__WASM_2X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__WASM_2X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__WASM_2X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PRELU__WASM_2X4, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, qmin) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmin(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, qmax) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .qmax(128)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4, PReLUMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // XNN_ARCH_WASM


TEST(F32_PRELU__SCALAR_2X1, channels_eq_1) {
  PReLUMicrokernelTester()
    .rows(2)
    .channels(1)
    .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
}

TEST(F32_PRELU__SCALAR_2X1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PRELU__SCALAR_2X1, rows_lt_2) {
  for (size_t rows = 1; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, rows_div_2) {
  for (size_t rows = 4; rows <= 8; rows += 2) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, rows_gt_2) {
  for (size_t rows = 3; rows < 4; rows++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, input_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(7)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, output_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .output_stride(7)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, inplace) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, qmin) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .qmin(128)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, qmax) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .qmax(128)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, channels_eq_4) {
  PReLUMicrokernelTester()
    .rows(2)
    .channels(4)
    .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
}

TEST(F32_PRELU__SCALAR_2X4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PRELU__SCALAR_2X4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PRELU__SCALAR_2X4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PRELU__SCALAR_2X4, rows_lt_2) {
  for (size_t rows = 1; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, rows_div_2) {
  for (size_t rows = 4; rows <= 8; rows += 2) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, rows_gt_2) {
  for (size_t rows = 3; rows < 4; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, input_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, output_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .output_stride(23)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, inplace) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, qmin) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .qmin(128)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, qmax) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .qmax(128)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4, PReLUMicrokernelTester::Variant::Scalar);
    }
  }
}