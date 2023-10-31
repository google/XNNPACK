// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-vhswish.yaml
//   Generator: tools/generate-vhswish-test.py


#include <gtest/gtest.h>

#include <vector>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vhswish.h>

#include "vhswish-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_VHSWISH__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VHSwishMicrokernelTester()
      .batch_size(8)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__neon_u8, xnn_init_qu8_hswish_neon_params);
  }

  TEST(QU8_VHSWISH__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__neon_u8, xnn_init_qu8_hswish_neon_params);
    }
  }

  TEST(QU8_VHSWISH__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__neon_u8, xnn_init_qu8_hswish_neon_params);
    }
  }

  TEST(QU8_VHSWISH__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__neon_u8, xnn_init_qu8_hswish_neon_params);
    }
  }

  TEST(QU8_VHSWISH__NEON_U8, input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__neon_u8, xnn_init_qu8_hswish_neon_params);
        }
    }
  }

  TEST(QU8_VHSWISH__NEON_U8, output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__neon_u8, xnn_init_qu8_hswish_neon_params);
        }
    }
  }

  TEST(QU8_VHSWISH__NEON_U8, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__neon_u8, xnn_init_qu8_hswish_neon_params);
      }
    }
  }

  TEST(QU8_VHSWISH__NEON_U8, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__neon_u8, xnn_init_qu8_hswish_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_VHSWISH__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VHSwishMicrokernelTester()
      .batch_size(16)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__neon_u16, xnn_init_qu8_hswish_neon_params);
  }

  TEST(QU8_VHSWISH__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__neon_u16, xnn_init_qu8_hswish_neon_params);
    }
  }

  TEST(QU8_VHSWISH__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__neon_u16, xnn_init_qu8_hswish_neon_params);
    }
  }

  TEST(QU8_VHSWISH__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__neon_u16, xnn_init_qu8_hswish_neon_params);
    }
  }

  TEST(QU8_VHSWISH__NEON_U16, input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__neon_u16, xnn_init_qu8_hswish_neon_params);
        }
    }
  }

  TEST(QU8_VHSWISH__NEON_U16, output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__neon_u16, xnn_init_qu8_hswish_neon_params);
        }
    }
  }

  TEST(QU8_VHSWISH__NEON_U16, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__neon_u16, xnn_init_qu8_hswish_neon_params);
      }
    }
  }

  TEST(QU8_VHSWISH__NEON_U16, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__neon_u16, xnn_init_qu8_hswish_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_VHSWISH__NEON_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VHSwishMicrokernelTester()
      .batch_size(32)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__neon_u32, xnn_init_qu8_hswish_neon_params);
  }

  TEST(QU8_VHSWISH__NEON_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__neon_u32, xnn_init_qu8_hswish_neon_params);
    }
  }

  TEST(QU8_VHSWISH__NEON_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__neon_u32, xnn_init_qu8_hswish_neon_params);
    }
  }

  TEST(QU8_VHSWISH__NEON_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__neon_u32, xnn_init_qu8_hswish_neon_params);
    }
  }

  TEST(QU8_VHSWISH__NEON_U32, input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__neon_u32, xnn_init_qu8_hswish_neon_params);
        }
    }
  }

  TEST(QU8_VHSWISH__NEON_U32, output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__neon_u32, xnn_init_qu8_hswish_neon_params);
        }
    }
  }

  TEST(QU8_VHSWISH__NEON_U32, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__neon_u32, xnn_init_qu8_hswish_neon_params);
      }
    }
  }

  TEST(QU8_VHSWISH__NEON_U32, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__neon_u32, xnn_init_qu8_hswish_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__SSE2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VHSwishMicrokernelTester()
      .batch_size(16)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__sse2_u16, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__SSE2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse2_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse2_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse2_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE2_U16, input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse2_u16, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE2_U16, output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse2_u16, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE2_U16, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse2_u16, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__SSE2_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__sse2_u16, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__SSE2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    VHSwishMicrokernelTester()
      .batch_size(32)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__sse2_u32, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__SSE2_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse2_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse2_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse2_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE2_U32, input_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse2_u32, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE2_U32, output_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse2_u32, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE2_U32, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse2_u32, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__SSE2_U32, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__sse2_u32, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__SSSE3_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSSE3;
    VHSwishMicrokernelTester()
      .batch_size(16)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__ssse3_u16, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__SSSE3_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__ssse3_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__ssse3_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__ssse3_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U16, input_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__ssse3_u16, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U16, output_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__ssse3_u16, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U16, input_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__ssse3_u16, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__ssse3_u16, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__SSSE3_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSSE3;
    VHSwishMicrokernelTester()
      .batch_size(32)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__ssse3_u32, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__SSSE3_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__ssse3_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__ssse3_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__ssse3_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U32, input_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__ssse3_u32, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U32, output_scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__ssse3_u32, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U32, input_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__ssse3_u32, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__SSSE3_U32, output_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__ssse3_u32, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__SSE41_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VHSwishMicrokernelTester()
      .batch_size(8)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__sse41_u8, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__SSE41_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse41_u8, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE41_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse41_u8, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE41_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse41_u8, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE41_U8, input_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u8, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE41_U8, output_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u8, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE41_U8, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u8, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__SSE41_U8, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u8, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__SSE41_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VHSwishMicrokernelTester()
      .batch_size(16)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__sse41_u16, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__SSE41_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse41_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE41_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse41_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE41_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse41_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE41_U16, input_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u16, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE41_U16, output_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u16, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE41_U16, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u16, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__SSE41_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u16, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__SSE41_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VHSwishMicrokernelTester()
      .batch_size(32)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__sse41_u32, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__SSE41_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse41_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE41_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse41_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE41_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__sse41_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__SSE41_U32, input_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u32, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE41_U32, output_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u32, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__SSE41_U32, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u32, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__SSE41_U32, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__sse41_u32, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VHSwishMicrokernelTester()
      .batch_size(8)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__avx_u8, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__avx_u8, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__avx_u8, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__avx_u8, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__AVX_U8, input_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__avx_u8, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__AVX_U8, output_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__avx_u8, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__AVX_U8, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__avx_u8, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__AVX_U8, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__avx_u8, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VHSwishMicrokernelTester()
      .batch_size(16)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__avx_u16, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__avx_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__avx_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__avx_u16, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__AVX_U16, input_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__avx_u16, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__AVX_U16, output_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__avx_u16, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__AVX_U16, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__avx_u16, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__AVX_U16, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__avx_u16, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_VHSWISH__AVX_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VHSwishMicrokernelTester()
      .batch_size(32)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__avx_u32, xnn_init_qu8_hswish_sse2_params);
  }

  TEST(QU8_VHSWISH__AVX_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__avx_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__AVX_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__avx_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__AVX_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__avx_u32, xnn_init_qu8_hswish_sse2_params);
    }
  }

  TEST(QU8_VHSWISH__AVX_U32, input_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__avx_u32, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__AVX_U32, output_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__avx_u32, xnn_init_qu8_hswish_sse2_params);
        }
    }
  }

  TEST(QU8_VHSWISH__AVX_U32, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__avx_u32, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }

  TEST(QU8_VHSWISH__AVX_U32, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__avx_u32, xnn_init_qu8_hswish_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QU8_VHSWISH__WASMSIMD_U8, batch_eq_8) {
    VHSwishMicrokernelTester()
      .batch_size(8)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u8, xnn_init_qu8_hswish_wasmsimd_params);
  }

  TEST(QU8_VHSWISH__WASMSIMD_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u8, xnn_init_qu8_hswish_wasmsimd_params);
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u8, xnn_init_qu8_hswish_wasmsimd_params);
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u8, xnn_init_qu8_hswish_wasmsimd_params);
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U8, input_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u8, xnn_init_qu8_hswish_wasmsimd_params);
        }
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U8, output_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u8, xnn_init_qu8_hswish_wasmsimd_params);
        }
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U8, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u8, xnn_init_qu8_hswish_wasmsimd_params);
      }
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U8, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u8, xnn_init_qu8_hswish_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QU8_VHSWISH__WASMSIMD_U16, batch_eq_16) {
    VHSwishMicrokernelTester()
      .batch_size(16)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u16, xnn_init_qu8_hswish_wasmsimd_params);
  }

  TEST(QU8_VHSWISH__WASMSIMD_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u16, xnn_init_qu8_hswish_wasmsimd_params);
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u16, xnn_init_qu8_hswish_wasmsimd_params);
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u16, xnn_init_qu8_hswish_wasmsimd_params);
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U16, input_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u16, xnn_init_qu8_hswish_wasmsimd_params);
        }
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U16, output_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u16, xnn_init_qu8_hswish_wasmsimd_params);
        }
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U16, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u16, xnn_init_qu8_hswish_wasmsimd_params);
      }
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U16, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u16, xnn_init_qu8_hswish_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QU8_VHSWISH__WASMSIMD_U32, batch_eq_32) {
    VHSwishMicrokernelTester()
      .batch_size(32)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u32, xnn_init_qu8_hswish_wasmsimd_params);
  }

  TEST(QU8_VHSWISH__WASMSIMD_U32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u32, xnn_init_qu8_hswish_wasmsimd_params);
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u32, xnn_init_qu8_hswish_wasmsimd_params);
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u32, xnn_init_qu8_hswish_wasmsimd_params);
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U32, input_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u32, xnn_init_qu8_hswish_wasmsimd_params);
        }
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U32, output_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .input_zero_point(150)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u32, xnn_init_qu8_hswish_wasmsimd_params);
        }
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U32, input_zero_point) {
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .output_zero_point(100)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u32, xnn_init_qu8_hswish_wasmsimd_params);
      }
    }
  }

  TEST(QU8_VHSWISH__WASMSIMD_U32, output_zero_point) {
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(150)
          .output_zero_point(output_zero_point)
          .Test(xnn_qu8_vhswish_ukernel__wasmsimd_u32, xnn_init_qu8_hswish_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(QU8_VHSWISH__SCALAR_U1, batch_eq_1) {
  VHSwishMicrokernelTester()
    .batch_size(1)
    .input_zero_point(150)
    .output_zero_point(100)
    .Test(xnn_qu8_vhswish_ukernel__scalar_u1, xnn_init_qu8_hswish_scalar_params);
}

TEST(QU8_VHSWISH__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VHSwishMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__scalar_u1, xnn_init_qu8_hswish_scalar_params);
  }
}

TEST(QU8_VHSWISH__SCALAR_U1, input_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_scale(input_scale)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u1, xnn_init_qu8_hswish_scalar_params);
      }
  }
}

TEST(QU8_VHSWISH__SCALAR_U1, output_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .output_scale(output_scale)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u1, xnn_init_qu8_hswish_scalar_params);
      }
  }
}

TEST(QU8_VHSWISH__SCALAR_U1, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u1, xnn_init_qu8_hswish_scalar_params);
    }
  }
}

TEST(QU8_VHSWISH__SCALAR_U1, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(output_zero_point)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u1, xnn_init_qu8_hswish_scalar_params);
    }
  }
}

TEST(QU8_VHSWISH__SCALAR_U2, batch_eq_2) {
  VHSwishMicrokernelTester()
    .batch_size(2)
    .input_zero_point(150)
    .output_zero_point(100)
    .Test(xnn_qu8_vhswish_ukernel__scalar_u2, xnn_init_qu8_hswish_scalar_params);
}

TEST(QU8_VHSWISH__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VHSwishMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__scalar_u2, xnn_init_qu8_hswish_scalar_params);
  }
}

TEST(QU8_VHSWISH__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VHSwishMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__scalar_u2, xnn_init_qu8_hswish_scalar_params);
  }
}

TEST(QU8_VHSWISH__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VHSwishMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__scalar_u2, xnn_init_qu8_hswish_scalar_params);
  }
}

TEST(QU8_VHSWISH__SCALAR_U2, input_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_scale(input_scale)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u2, xnn_init_qu8_hswish_scalar_params);
      }
  }
}

TEST(QU8_VHSWISH__SCALAR_U2, output_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .output_scale(output_scale)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u2, xnn_init_qu8_hswish_scalar_params);
      }
  }
}

TEST(QU8_VHSWISH__SCALAR_U2, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u2, xnn_init_qu8_hswish_scalar_params);
    }
  }
}

TEST(QU8_VHSWISH__SCALAR_U2, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(output_zero_point)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u2, xnn_init_qu8_hswish_scalar_params);
    }
  }
}

TEST(QU8_VHSWISH__SCALAR_U4, batch_eq_4) {
  VHSwishMicrokernelTester()
    .batch_size(4)
    .input_zero_point(150)
    .output_zero_point(100)
    .Test(xnn_qu8_vhswish_ukernel__scalar_u4, xnn_init_qu8_hswish_scalar_params);
}

TEST(QU8_VHSWISH__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VHSwishMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__scalar_u4, xnn_init_qu8_hswish_scalar_params);
  }
}

TEST(QU8_VHSWISH__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VHSwishMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__scalar_u4, xnn_init_qu8_hswish_scalar_params);
  }
}

TEST(QU8_VHSWISH__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VHSwishMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(150)
      .output_zero_point(100)
      .Test(xnn_qu8_vhswish_ukernel__scalar_u4, xnn_init_qu8_hswish_scalar_params);
  }
}

TEST(QU8_VHSWISH__SCALAR_U4, input_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_scale(input_scale)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u4, xnn_init_qu8_hswish_scalar_params);
      }
  }
}

TEST(QU8_VHSWISH__SCALAR_U4, output_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .output_scale(output_scale)
        .input_zero_point(150)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u4, xnn_init_qu8_hswish_scalar_params);
      }
  }
}

TEST(QU8_VHSWISH__SCALAR_U4, input_zero_point) {
  for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .output_zero_point(100)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u4, xnn_init_qu8_hswish_scalar_params);
    }
  }
}

TEST(QU8_VHSWISH__SCALAR_U4, output_zero_point) {
  for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(150)
        .output_zero_point(output_zero_point)
        .Test(xnn_qu8_vhswish_ukernel__scalar_u4, xnn_init_qu8_hswish_scalar_params);
    }
  }
}