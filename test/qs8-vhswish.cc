// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-vhswish.yaml
//   Generator: tools/generate-vhswish-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vhswish.h>
#include "vhswish-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VHSWISH__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VHSwishMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vhswish_ukernel__neon_x8, xnn_init_qs8_hswish_neon_params);
  }

  TEST(QS8_VHSWISH__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vhswish_ukernel__neon_x8, xnn_init_qs8_hswish_neon_params);
    }
  }

  TEST(QS8_VHSWISH__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vhswish_ukernel__neon_x8, xnn_init_qs8_hswish_neon_params);
    }
  }

  TEST(QS8_VHSWISH__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vhswish_ukernel__neon_x8, xnn_init_qs8_hswish_neon_params);
    }
  }

  TEST(QS8_VHSWISH__NEON_X8, input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .Test(xnn_qs8_vhswish_ukernel__neon_x8, xnn_init_qs8_hswish_neon_params);
        }
    }
  }

  TEST(QS8_VHSWISH__NEON_X8, output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .Test(xnn_qs8_vhswish_ukernel__neon_x8, xnn_init_qs8_hswish_neon_params);
        }
    }
  }

  TEST(QS8_VHSWISH__NEON_X8, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vhswish_ukernel__neon_x8, xnn_init_qs8_hswish_neon_params);
      }
    }
  }

  TEST(QS8_VHSWISH__NEON_X8, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vhswish_ukernel__neon_x8, xnn_init_qs8_hswish_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VHSWISH__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VHSwishMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vhswish_ukernel__neon_x16, xnn_init_qs8_hswish_neon_params);
  }

  TEST(QS8_VHSWISH__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vhswish_ukernel__neon_x16, xnn_init_qs8_hswish_neon_params);
    }
  }

  TEST(QS8_VHSWISH__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vhswish_ukernel__neon_x16, xnn_init_qs8_hswish_neon_params);
    }
  }

  TEST(QS8_VHSWISH__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vhswish_ukernel__neon_x16, xnn_init_qs8_hswish_neon_params);
    }
  }

  TEST(QS8_VHSWISH__NEON_X16, input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .Test(xnn_qs8_vhswish_ukernel__neon_x16, xnn_init_qs8_hswish_neon_params);
        }
    }
  }

  TEST(QS8_VHSWISH__NEON_X16, output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .Test(xnn_qs8_vhswish_ukernel__neon_x16, xnn_init_qs8_hswish_neon_params);
        }
    }
  }

  TEST(QS8_VHSWISH__NEON_X16, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vhswish_ukernel__neon_x16, xnn_init_qs8_hswish_neon_params);
      }
    }
  }

  TEST(QS8_VHSWISH__NEON_X16, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vhswish_ukernel__neon_x16, xnn_init_qs8_hswish_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VHSWISH__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VHSwishMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vhswish_ukernel__neon_x32, xnn_init_qs8_hswish_neon_params);
  }

  TEST(QS8_VHSWISH__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vhswish_ukernel__neon_x32, xnn_init_qs8_hswish_neon_params);
    }
  }

  TEST(QS8_VHSWISH__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vhswish_ukernel__neon_x32, xnn_init_qs8_hswish_neon_params);
    }
  }

  TEST(QS8_VHSWISH__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VHSwishMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vhswish_ukernel__neon_x32, xnn_init_qs8_hswish_neon_params);
    }
  }

  TEST(QS8_VHSWISH__NEON_X32, input_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float input_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_scale(input_scale)
          .Test(xnn_qs8_vhswish_ukernel__neon_x32, xnn_init_qs8_hswish_neon_params);
        }
    }
  }

  TEST(QS8_VHSWISH__NEON_X32, output_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float output_scale : std::vector<float>({4.0f, 16.0f, 64.0f})) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_scale(output_scale)
          .Test(xnn_qs8_vhswish_ukernel__neon_x32, xnn_init_qs8_hswish_neon_params);
        }
    }
  }

  TEST(QS8_VHSWISH__NEON_X32, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 2; input_zero_point < 10; input_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_vhswish_ukernel__neon_x32, xnn_init_qs8_hswish_neon_params);
      }
    }
  }

  TEST(QS8_VHSWISH__NEON_X32, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 2; output_zero_point < 10; output_zero_point += 3) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VHSwishMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .Test(xnn_qs8_vhswish_ukernel__neon_x32, xnn_init_qs8_hswish_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
