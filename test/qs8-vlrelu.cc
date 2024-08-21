// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-vlrelu.yaml
//   Generator: tools/generate-vlrelu-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vlrelu.h"
#include "vlrelu-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VLRELU__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__neon_u8, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__NEON_U8, positive_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_U8, negative_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_U8, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__NEON_U8, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VLRELU__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__neon_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__NEON_U16, positive_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_U16, negative_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_U16, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__NEON_U16, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VLRELU__NEON_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__neon_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__NEON_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__NEON_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__NEON_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__neon_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__NEON_U32, positive_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_U32, negative_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__NEON_U32, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__NEON_U32, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__neon_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__sse2_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__SSE2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_U16, positive_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE2_U16, negative_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE2_U16, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE2_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__sse2_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__SSE2_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse2_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE2_U32, positive_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE2_U32, negative_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE2_U32, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE2_U32, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse2_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSSE3_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSSE3;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__ssse3_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__SSSE3_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_U16, positive_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSSE3_U16, negative_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSSE3_U16, input_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSSE3_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSSE3_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSSE3;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__ssse3_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__SSSE3_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__ssse3_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSSE3_U32, positive_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSSE3_U32, negative_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSSE3_U32, input_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSSE3_U32, output_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__ssse3_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE41_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__sse41_u8, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__SSE41_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_U8, positive_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_U8, negative_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_U8, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE41_U8, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE41_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__sse41_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__SSE41_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_U16, positive_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_U16, negative_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_U16, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE41_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__SSE41_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__sse41_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__SSE41_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__sse41_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__SSE41_U32, positive_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_U32, negative_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__SSE41_U32, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__SSE41_U32, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__sse41_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__avx_u8, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX_U8, positive_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_U8, negative_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_U8, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX_U8, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__avx_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX_U16, positive_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_U16, negative_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_U16, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX_U16, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__avx_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__AVX_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX_U32, positive_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_U32, negative_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX_U32, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX_U32, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__avx2_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__AVX2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_U16, positive_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_U16, negative_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_U16, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX2_U16, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__avx2_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__AVX2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_U32, positive_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_U32, negative_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_U32, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX2_U32, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VLRELU__AVX2_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VLReLUMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs8_vlrelu_ukernel__avx2_u64, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__AVX2_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_u64, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_u64, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__avx2_u64, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__AVX2_U64, positive_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u64, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_U64, negative_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u64, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__AVX2_U64, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u64, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__AVX2_U64, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__avx2_u64, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_ARM_U16, batch_eq_16) {
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U16, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U16, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U16, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U16, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_ARM_U32, batch_eq_32) {
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U32, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U32, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U32, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_ARM_U32, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_X86_U8, batch_eq_8) {
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U8, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U8, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U8, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U8, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_X86_U16, batch_eq_16) {
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U16, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U16, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U16, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U16, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMSIMD_X86_U32, batch_eq_32) {
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U32, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U32, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U32, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMSIMD_X86_U32, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U16, batch_eq_16) {
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U16, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U16, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U16, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U16, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U32, batch_eq_32) {
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U32, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U32, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U32, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_ARM_U32, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U8, batch_eq_8) {
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U8, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U8, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U8, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U8, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U16, batch_eq_16) {
    VLReLUMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U16, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U16, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U16, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U16, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U32, batch_eq_32) {
    VLReLUMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U32, positive_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U32, negative_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U32, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__WASMRELAXEDSIMD_X86_U32, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM
  TEST(QS8_VLRELU__ARMSIMD32_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_SIMD32;
    VLReLUMicrokernelTester()
      .batch_size(4)
      .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u4, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__ARMSIMD32_U4, batch_div_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u4, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u4, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u4, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U4, positive_scale) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u4, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U4, negative_scale) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u4, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U4, input_zero_point) {
    TEST_REQUIRES_ARM_SIMD32;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u4, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U4, output_zero_point) {
    TEST_REQUIRES_ARM_SIMD32;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u4, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM
  TEST(QS8_VLRELU__ARMSIMD32_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_SIMD32;
    VLReLUMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u8, xnn_init_qs8_lrelu_scalar_params);
  }

  TEST(QS8_VLRELU__ARMSIMD32_U8, batch_div_8) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u8, xnn_init_qs8_lrelu_scalar_params);
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U8, positive_scale) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .positive_scale(positive_scale)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U8, negative_scale) {
    TEST_REQUIRES_ARM_SIMD32;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .negative_scale(negative_scale)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u8, xnn_init_qs8_lrelu_scalar_params);
        }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U8, input_zero_point) {
    TEST_REQUIRES_ARM_SIMD32;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }

  TEST(QS8_VLRELU__ARMSIMD32_U8, output_zero_point) {
    TEST_REQUIRES_ARM_SIMD32;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VLReLUMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vlrelu_ukernel__armsimd32_u8, xnn_init_qs8_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM


TEST(QS8_VLRELU__SCALAR_SELECT_U1, batch_eq_1) {
  VLReLUMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u1, xnn_init_qs8_lrelu_scalar_params);
}

TEST(QS8_VLRELU__SCALAR_SELECT_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u1, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U1, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u1, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U1, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u1, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U1, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u1, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U1, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u1, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U2, batch_eq_2) {
  VLReLUMicrokernelTester()
    .batch_size(2)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u2, xnn_init_qs8_lrelu_scalar_params);
}

TEST(QS8_VLRELU__SCALAR_SELECT_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u2, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u2, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u2, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U2, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u2, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U2, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u2, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U2, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u2, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U2, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u2, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U4, batch_eq_4) {
  VLReLUMicrokernelTester()
    .batch_size(4)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u4, xnn_init_qs8_lrelu_scalar_params);
}

TEST(QS8_VLRELU__SCALAR_SELECT_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u4, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u4, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u4, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U4, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u4, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U4, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u4, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U4, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u4, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_SELECT_U4, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_select_u4, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U1, batch_eq_1) {
  VLReLUMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u1, xnn_init_qs8_lrelu_scalar_params);
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u1, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U1, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u1, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U1, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u1, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U1, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u1, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U1, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u1, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U2, batch_eq_2) {
  VLReLUMicrokernelTester()
    .batch_size(2)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u2, xnn_init_qs8_lrelu_scalar_params);
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u2, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u2, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u2, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U2, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u2, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U2, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u2, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U2, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u2, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U2, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u2, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U4, batch_eq_4) {
  VLReLUMicrokernelTester()
    .batch_size(4)
    .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u4, xnn_init_qs8_lrelu_scalar_params);
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u4, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u4, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VLReLUMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u4, xnn_init_qs8_lrelu_scalar_params);
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U4, positive_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .positive_scale(positive_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u4, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U4, negative_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .negative_scale(negative_scale)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u4, xnn_init_qs8_lrelu_scalar_params);
      }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U4, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u4, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}

TEST(QS8_VLRELU__SCALAR_ANDXOR_U4, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VLReLUMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .Test(xnn_qs8_vlrelu_ukernel__scalar_andxor_u4, xnn_init_qs8_lrelu_scalar_params);
    }
  }
}