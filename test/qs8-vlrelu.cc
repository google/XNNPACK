// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-vlrelu.yaml
//   Generator: tools/generate-vlrelu-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vlrelu.h>
#include "vlrelu-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VLRELU__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__neon_x8, xnn_init_qs8_lrelu_neon_params);
  }

  TEST(QS8_VLRELU__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_x8, xnn_init_qs8_lrelu_neon_params);
    }
  }

  TEST(QS8_VLRELU__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_x8, xnn_init_qs8_lrelu_neon_params);
    }
  }

  TEST(QS8_VLRELU__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_x8, xnn_init_qs8_lrelu_neon_params);
    }
  }

  TEST(QS8_VLRELU__NEON_X8, positive_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x8, xnn_init_qs8_lrelu_neon_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_X8, negative_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x8, xnn_init_qs8_lrelu_neon_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_X8, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x8, xnn_init_qs8_lrelu_neon_params);
      }
    }
  }

  TEST(QS8_VLRELU__NEON_X8, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x8, xnn_init_qs8_lrelu_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VLRELU__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__neon_x16, xnn_init_qs8_lrelu_neon_params);
  }

  TEST(QS8_VLRELU__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_x16, xnn_init_qs8_lrelu_neon_params);
    }
  }

  TEST(QS8_VLRELU__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_x16, xnn_init_qs8_lrelu_neon_params);
    }
  }

  TEST(QS8_VLRELU__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_x16, xnn_init_qs8_lrelu_neon_params);
    }
  }

  TEST(QS8_VLRELU__NEON_X16, positive_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x16, xnn_init_qs8_lrelu_neon_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_X16, negative_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x16, xnn_init_qs8_lrelu_neon_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_X16, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x16, xnn_init_qs8_lrelu_neon_params);
      }
    }
  }

  TEST(QS8_VLRELU__NEON_X16, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x16, xnn_init_qs8_lrelu_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VLRELU__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__neon_x32, xnn_init_qs8_lrelu_neon_params);
  }

  TEST(QS8_VLRELU__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_x32, xnn_init_qs8_lrelu_neon_params);
    }
  }

  TEST(QS8_VLRELU__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_x32, xnn_init_qs8_lrelu_neon_params);
    }
  }

  TEST(QS8_VLRELU__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_x32, xnn_init_qs8_lrelu_neon_params);
    }
  }

  TEST(QS8_VLRELU__NEON_X32, positive_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x32, xnn_init_qs8_lrelu_neon_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_X32, negative_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x32, xnn_init_qs8_lrelu_neon_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_X32, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x32, xnn_init_qs8_lrelu_neon_params);
      }
    }
  }

  TEST(QS8_VLRELU__NEON_X32, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_x32, xnn_init_qs8_lrelu_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__sse2_x16, xnn_init_qs8_lrelu_sse2_params);
  }

  TEST(QS8_VLRELU__SSE2_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_x16, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_x16, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_x16, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_X16, positive_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_x16, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE2_X16, negative_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_x16, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE2_X16, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_x16, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE2_X16, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_x16, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE2_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__sse2_x32, xnn_init_qs8_lrelu_sse2_params);
  }

  TEST(QS8_VLRELU__SSE2_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_x32, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_x32, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_x32, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_X32, positive_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_x32, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE2_X32, negative_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_x32, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE2_X32, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_x32, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE2_X32, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_x32, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSSE3_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSSE3;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__ssse3_x16, xnn_init_qs8_lrelu_sse2_params);
  }

  TEST(QS8_VLRELU__SSSE3_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_x16, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_x16, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_x16, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_X16, positive_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_x16, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSSE3_X16, negative_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_x16, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSSE3_X16, input_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_x16, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSSE3_X16, output_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_x16, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSSE3_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSSE3;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__ssse3_x32, xnn_init_qs8_lrelu_sse2_params);
  }

  TEST(QS8_VLRELU__SSSE3_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_x32, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_x32, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_x32, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_X32, positive_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_x32, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSSE3_X32, negative_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_x32, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSSE3_X32, input_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_x32, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSSE3_X32, output_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_x32, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE41_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__sse41_x8, xnn_init_qs8_lrelu_sse2_params);
  }

  TEST(QS8_VLRELU__SSE41_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_x8, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_x8, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_x8, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_X8, positive_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x8, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_X8, negative_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x8, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_X8, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x8, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE41_X8, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x8, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE41_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__sse41_x16, xnn_init_qs8_lrelu_sse2_params);
  }

  TEST(QS8_VLRELU__SSE41_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_x16, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_x16, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_x16, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_X16, positive_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x16, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_X16, negative_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x16, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_X16, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x16, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE41_X16, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x16, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE41_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__sse41_x32, xnn_init_qs8_lrelu_sse2_params);
  }

  TEST(QS8_VLRELU__SSE41_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_x32, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_x32, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_x32, xnn_init_qs8_lrelu_sse2_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_X32, positive_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x32, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_X32, negative_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x32, xnn_init_qs8_lrelu_sse2_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_X32, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x32, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE41_X32, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_x32, xnn_init_qs8_lrelu_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__avx_x8, xnn_init_qs8_lrelu_avx_params);
  }

  TEST(QS8_VLRELU__AVX_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_x8, xnn_init_qs8_lrelu_avx_params);
    }
  }

  TEST(QS8_VLRELU__AVX_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_x8, xnn_init_qs8_lrelu_avx_params);
    }
  }

  TEST(QS8_VLRELU__AVX_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_x8, xnn_init_qs8_lrelu_avx_params);
    }
  }

  TEST(QS8_VLRELU__AVX_X8, positive_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x8, xnn_init_qs8_lrelu_avx_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_X8, negative_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x8, xnn_init_qs8_lrelu_avx_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_X8, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x8, xnn_init_qs8_lrelu_avx_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX_X8, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x8, xnn_init_qs8_lrelu_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__avx_x16, xnn_init_qs8_lrelu_avx_params);
  }

  TEST(QS8_VLRELU__AVX_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_x16, xnn_init_qs8_lrelu_avx_params);
    }
  }

  TEST(QS8_VLRELU__AVX_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_x16, xnn_init_qs8_lrelu_avx_params);
    }
  }

  TEST(QS8_VLRELU__AVX_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_x16, xnn_init_qs8_lrelu_avx_params);
    }
  }

  TEST(QS8_VLRELU__AVX_X16, positive_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x16, xnn_init_qs8_lrelu_avx_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_X16, negative_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x16, xnn_init_qs8_lrelu_avx_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_X16, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x16, xnn_init_qs8_lrelu_avx_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX_X16, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x16, xnn_init_qs8_lrelu_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__avx_x32, xnn_init_qs8_lrelu_avx_params);
  }

  TEST(QS8_VLRELU__AVX_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_x32, xnn_init_qs8_lrelu_avx_params);
    }
  }

  TEST(QS8_VLRELU__AVX_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_x32, xnn_init_qs8_lrelu_avx_params);
    }
  }

  TEST(QS8_VLRELU__AVX_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_x32, xnn_init_qs8_lrelu_avx_params);
    }
  }

  TEST(QS8_VLRELU__AVX_X32, positive_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x32, xnn_init_qs8_lrelu_avx_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_X32, negative_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x32, xnn_init_qs8_lrelu_avx_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_X32, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x32, xnn_init_qs8_lrelu_avx_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX_X32, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_x32, xnn_init_qs8_lrelu_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__avx2_x16, xnn_init_qs8_lrelu_avx2_params);
  }

  TEST(QS8_VLRELU__AVX2_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_x16, xnn_init_qs8_lrelu_avx2_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_x16, xnn_init_qs8_lrelu_avx2_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_x16, xnn_init_qs8_lrelu_avx2_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_X16, positive_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x16, xnn_init_qs8_lrelu_avx2_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_X16, negative_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x16, xnn_init_qs8_lrelu_avx2_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_X16, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x16, xnn_init_qs8_lrelu_avx2_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX2_X16, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x16, xnn_init_qs8_lrelu_avx2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX2_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__avx2_x32, xnn_init_qs8_lrelu_avx2_params);
  }

  TEST(QS8_VLRELU__AVX2_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_x32, xnn_init_qs8_lrelu_avx2_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_x32, xnn_init_qs8_lrelu_avx2_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_x32, xnn_init_qs8_lrelu_avx2_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_X32, positive_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x32, xnn_init_qs8_lrelu_avx2_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_X32, negative_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x32, xnn_init_qs8_lrelu_avx2_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_X32, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x32, xnn_init_qs8_lrelu_avx2_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX2_X32, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x32, xnn_init_qs8_lrelu_avx2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX2_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VLReLUMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs8_vlrelu_ukernel__avx2_x64, xnn_init_qs8_lrelu_avx2_params);
  }

  TEST(QS8_VLRELU__AVX2_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_x64, xnn_init_qs8_lrelu_avx2_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_x64, xnn_init_qs8_lrelu_avx2_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_x64, xnn_init_qs8_lrelu_avx2_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_X64, positive_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x64, xnn_init_qs8_lrelu_avx2_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_X64, negative_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x64, xnn_init_qs8_lrelu_avx2_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_X64, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x64, xnn_init_qs8_lrelu_avx2_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX2_X64, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_x64, xnn_init_qs8_lrelu_avx2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_ARM_X16, batch_eq_16) {
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X16, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X16, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X16, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X16, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_ARM_X32, batch_eq_32) {
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X32, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X32, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X32, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_X32, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_X86_X8, batch_eq_8) {
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X8, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X8, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X8, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X8, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_X86_X16, batch_eq_16) {
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X16, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X16, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X16, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X16, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_X86_X32, batch_eq_32) {
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X32, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X32, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X32, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_X32, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X16, batch_eq_16) {
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X16, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X16, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X16, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X16, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x16, xnn_init_qs8_lrelu_wasmsimd_arm_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X32, batch_eq_32) {
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X32, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X32, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X32, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_X32, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_x32, xnn_init_qs8_lrelu_wasmsimd_arm_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X8, batch_eq_8) {
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X8, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X8, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X8, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X8, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x8, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X16, batch_eq_16) {
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X16, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X16, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X16, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X16, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x16, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X32, batch_eq_32) {
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X32, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X32, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X32, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_X32, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_x32, xnn_init_qs8_lrelu_wasmsimd_x86_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM
  TEST(QS8_VLRELU__ARMSIMD32_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_SIMD32;
    VLReLUMicrokernelTester()
      .batch_size(4)
      .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x4, xnn_init_qs8_lrelu_armsimd32_params);
  }

  TEST(QS8_VLRELU__ARMSIMD32_X4, batch_div_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x4, xnn_init_qs8_lrelu_armsimd32_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x4, xnn_init_qs8_lrelu_armsimd32_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x4, xnn_init_qs8_lrelu_armsimd32_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X4, positive_scale) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x4, xnn_init_qs8_lrelu_armsimd32_params);
        }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X4, negative_scale) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x4, xnn_init_qs8_lrelu_armsimd32_params);
        }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X4, input_zero_point) {
    TEST_REQUIRES_ARM_SIMD32;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x4, xnn_init_qs8_lrelu_armsimd32_params);
      }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X4, output_zero_point) {
    TEST_REQUIRES_ARM_SIMD32;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x4, xnn_init_qs8_lrelu_armsimd32_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM
  TEST(QS8_VLRELU__ARMSIMD32_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_SIMD32;
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x8, xnn_init_qs8_lrelu_armsimd32_params);
  }

  TEST(QS8_VLRELU__ARMSIMD32_X8, batch_div_8) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x8, xnn_init_qs8_lrelu_armsimd32_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x8, xnn_init_qs8_lrelu_armsimd32_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x8, xnn_init_qs8_lrelu_armsimd32_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X8, positive_scale) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x8, xnn_init_qs8_lrelu_armsimd32_params);
        }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X8, negative_scale) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x8, xnn_init_qs8_lrelu_armsimd32_params);
        }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X8, input_zero_point) {
    TEST_REQUIRES_ARM_SIMD32;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x8, xnn_init_qs8_lrelu_armsimd32_params);
      }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_X8, output_zero_point) {
    TEST_REQUIRES_ARM_SIMD32;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_x8, xnn_init_qs8_lrelu_armsimd32_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM


TEST(QS8_VLRELU__SCALAR_SELECT_X1, batch_eq_1) {
  VLReLUMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x1, xnn_init_qs8_lrelu_scalar_select_params);
}

TEST(QS8_VLRELU__SCALAR_SELECT_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x1, xnn_init_qs8_lrelu_scalar_select_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X1, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x1, xnn_init_qs8_lrelu_scalar_select_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X1, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x1, xnn_init_qs8_lrelu_scalar_select_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X1, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x1, xnn_init_qs8_lrelu_scalar_select_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X1, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x1, xnn_init_qs8_lrelu_scalar_select_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X2, batch_eq_2) {
  VLReLUMicrokernelTester()
    .batch_size(2)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x2, xnn_init_qs8_lrelu_scalar_select_params);
}

TEST(QS8_VLRELU__SCALAR_SELECT_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x2, xnn_init_qs8_lrelu_scalar_select_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x2, xnn_init_qs8_lrelu_scalar_select_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x2, xnn_init_qs8_lrelu_scalar_select_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X2, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x2, xnn_init_qs8_lrelu_scalar_select_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X2, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x2, xnn_init_qs8_lrelu_scalar_select_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X2, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x2, xnn_init_qs8_lrelu_scalar_select_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X2, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x2, xnn_init_qs8_lrelu_scalar_select_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X4, batch_eq_4) {
  VLReLUMicrokernelTester()
    .batch_size(4)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x4, xnn_init_qs8_lrelu_scalar_select_params);
}

TEST(QS8_VLRELU__SCALAR_SELECT_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x4, xnn_init_qs8_lrelu_scalar_select_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x4, xnn_init_qs8_lrelu_scalar_select_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x4, xnn_init_qs8_lrelu_scalar_select_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X4, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x4, xnn_init_qs8_lrelu_scalar_select_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X4, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x4, xnn_init_qs8_lrelu_scalar_select_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X4, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x4, xnn_init_qs8_lrelu_scalar_select_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_X4, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_x4, xnn_init_qs8_lrelu_scalar_select_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X1, batch_eq_1) {
  VLReLUMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x1, xnn_init_qs8_lrelu_scalar_andxor_params);
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x1, xnn_init_qs8_lrelu_scalar_andxor_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X1, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x1, xnn_init_qs8_lrelu_scalar_andxor_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X1, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x1, xnn_init_qs8_lrelu_scalar_andxor_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X1, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x1, xnn_init_qs8_lrelu_scalar_andxor_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X1, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x1, xnn_init_qs8_lrelu_scalar_andxor_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X2, batch_eq_2) {
  VLReLUMicrokernelTester()
    .batch_size(2)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x2, xnn_init_qs8_lrelu_scalar_andxor_params);
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x2, xnn_init_qs8_lrelu_scalar_andxor_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x2, xnn_init_qs8_lrelu_scalar_andxor_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x2, xnn_init_qs8_lrelu_scalar_andxor_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X2, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x2, xnn_init_qs8_lrelu_scalar_andxor_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X2, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x2, xnn_init_qs8_lrelu_scalar_andxor_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X2, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x2, xnn_init_qs8_lrelu_scalar_andxor_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X2, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x2, xnn_init_qs8_lrelu_scalar_andxor_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X4, batch_eq_4) {
  VLReLUMicrokernelTester()
    .batch_size(4)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x4, xnn_init_qs8_lrelu_scalar_andxor_params);
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x4, xnn_init_qs8_lrelu_scalar_andxor_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x4, xnn_init_qs8_lrelu_scalar_andxor_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x4, xnn_init_qs8_lrelu_scalar_andxor_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X4, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x4, xnn_init_qs8_lrelu_scalar_andxor_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X4, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x4, xnn_init_qs8_lrelu_scalar_andxor_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X4, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x4, xnn_init_qs8_lrelu_scalar_andxor_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_X4, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_x4, xnn_init_qs8_lrelu_scalar_andxor_params);
    }
  }
}