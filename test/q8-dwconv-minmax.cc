// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/q8-dwconv-minmax.yaml
//   Generator: tools/generate-dwconv-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_ARM
  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
      }
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, input_zero_point_only) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .input_zero_point(255)
        .kernel_zero_point(0)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__AARCH32_NEON, kernel_zero_point_only) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .input_zero_point(0)
        .kernel_zero_point(255)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon);
    }
  }
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
      }
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, input_zero_point_only) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .input_zero_point(255)
        .kernel_zero_point(0)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__NEON, kernel_zero_point_only) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .input_zero_point(0)
        .kernel_zero_point(255)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__neon);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, c_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, c_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, c_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, c_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, multipixel) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
      }
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, input_zero_point_only) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .input_zero_point(255)
        .kernel_zero_point(0)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }

  TEST(Q8_DWCONV_MINMAX_UP8X9__SSE2, kernel_zero_point_only) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .input_zero_point(0)
        .kernel_zero_point(255)
        .Test(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(9)
    .channels(1)
    .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, input_zero_point_only) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .input_zero_point(255)
      .kernel_zero_point(0)
      .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_DWCONV_MINMAX_UP1X9__SCALAR, kernel_zero_point_only) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .input_zero_point(0)
      .kernel_zero_point(255)
      .Test(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}