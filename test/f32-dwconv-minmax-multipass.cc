// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-dwconv-minmax-multipass.yaml
//   Generator: tools/generate-dwconv-multipass-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_eq_8_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_eq_8_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_eq_4_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_eq_4_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_eq_8_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_eq_8_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_eq_4_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_eq_4_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_eq_8_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_ARM_NEON;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_ARM_NEON;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_eq_8_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_ARM_NEON_FMA;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_ARM_NEON_FMA;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_eq_8_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_div_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_gt_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_eq_16_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_div_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_gt_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(16)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_eq_16_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_div_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_gt_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(16)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_eq_4_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L4C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_eq_8_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_div_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_gt_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_eq_16_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_div_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_gt_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(16)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(7)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(13)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_eq_16_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_div_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_gt_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(16)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_eq_4_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L4C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_eq_8_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_div_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_gt_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_eq_16_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_div_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_gt_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(16)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(17)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_eq_16_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_div_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_gt_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(16)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(6)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(10)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_eq_8_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_div_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_gt_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(8)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(6)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(10)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(8)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(6)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(10)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_eq_16_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_div_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_gt_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(16)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(6)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(10)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_eq_16_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_div_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_gt_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(16)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(7)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(13)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_eq_8_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_div_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_gt_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(8)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(7)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(13)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(8)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L8C8S4R__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(7)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(13)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_eq_16_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_div_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_gt_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(16)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(7)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(13)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_eq_16_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_div_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_gt_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(16)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L16C8S4R__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(17)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_eq_8_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_div_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_gt_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(8)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(17)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(8)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L8C8S4R__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(17)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_eq_16_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_div_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_gt_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(16)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(17)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_eq_16_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      TEST_REQUIRES_X86_AVX;
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_div_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_gt_16_multipass) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        TEST_REQUIRES_X86_AVX;
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(16)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L16C8S4R__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(6)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(10)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_eq_8_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_FMA3;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_div_8_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_gt_8_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_FMA3;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(8)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L8C8S4R__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(6)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(10)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_eq_16_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_FMA3;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_div_16_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_gt_16_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_FMA3;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(16)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C8S4R__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_eq_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(32)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(6)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_eq_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(32)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(10)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_eq_32_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_FMA3;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(32)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_div_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_div_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_div_32_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_gt_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_gt_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_gt_32_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_eq_32_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_eq_32_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, c_eq_32_multipass_multipixel) {
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_FMA3;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(32)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(163)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C8S4R__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(592)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(7)
      .middle_pass_tile(6)
      .last_pass_tile(6)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(8)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(7)
      .middle_pass_tile(6)
      .last_pass_tile(6)
      .channel_tile(8)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(13)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_eq_8_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_FMA3;
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(8)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_div_8_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(8)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_gt_8_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(8)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(8)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_FMA3;
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 7; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(7)
            .middle_pass_tile(6)
            .last_pass_tile(6)
            .channel_tile(8)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L8C8S4R__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(8)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(7)
      .middle_pass_tile(6)
      .last_pass_tile(6)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(8)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(7)
      .middle_pass_tile(6)
      .last_pass_tile(6)
      .channel_tile(16)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(13)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_eq_16_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_FMA3;
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(8)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_div_16_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(8)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_gt_16_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(8)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(16)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_FMA3;
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 7; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(7)
            .middle_pass_tile(6)
            .last_pass_tile(6)
            .channel_tile(16)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L16C8S4R__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(16)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_eq_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(7)
      .middle_pass_tile(6)
      .last_pass_tile(6)
      .channel_tile(32)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(8)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_eq_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .first_pass_tile(7)
      .middle_pass_tile(6)
      .last_pass_tile(6)
      .channel_tile(32)
      .channel_subtile(8)
      .channel_round(4)
      .kernel_size(13)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_eq_32_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      TEST_REQUIRES_X86_FMA3;
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(32)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_div_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(8)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_div_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_div_32_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_gt_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(8)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_gt_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_gt_32_multipass) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_eq_32_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(8)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_eq_32_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .first_pass_tile(7)
        .middle_pass_tile(6)
        .last_pass_tile(6)
        .channel_tile(32)
        .channel_subtile(8)
        .channel_round(4)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, c_eq_32_multipass_multipixel) {
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        TEST_REQUIRES_X86_FMA3;
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 7; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(7)
            .middle_pass_tile(6)
            .last_pass_tile(6)
            .channel_tile(32)
            .channel_subtile(8)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(163)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_7F6M6L32C8S4R__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(7)
          .middle_pass_tile(6)
          .last_pass_tile(6)
          .channel_tile(32)
          .channel_subtile(8)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(592)
          .Test(xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(16)
      .channel_round(1)
      .kernel_size(6)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(16)
      .channel_round(1)
      .kernel_size(10)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_eq_16_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_AVX512F;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_div_16_multipass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_gt_16_multipass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_AVX512F;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(16)
            .channel_subtile(16)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(16)
      .channel_round(1)
      .kernel_size(6)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(16)
      .channel_subtile(16)
      .channel_round(1)
      .kernel_size(10)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_eq_16_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_AVX512F;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_div_16_multipass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_gt_16_multipass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(16)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_AVX512F;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(16)
            .channel_subtile(16)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L16C16S1R__AVX512F_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(16)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_eq_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(32)
      .channel_subtile(16)
      .channel_round(1)
      .kernel_size(6)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_eq_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(32)
      .channel_subtile(16)
      .channel_round(1)
      .kernel_size(10)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_eq_32_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_AVX512F;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(32)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_div_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_div_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_div_32_multipass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_gt_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_gt_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_gt_32_multipass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_eq_32_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_eq_32_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, c_eq_32_multipass_multipixel) {
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_AVX512F;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(32)
            .channel_subtile(16)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(163)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(592)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_eq_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(32)
      .channel_subtile(16)
      .channel_round(1)
      .kernel_size(6)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_eq_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(32)
      .channel_subtile(16)
      .channel_round(1)
      .kernel_size(10)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_eq_32_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      TEST_REQUIRES_X86_AVX512F;
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(32)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_div_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_div_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_div_32_multipass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_gt_32_first_pass_plus_one) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_gt_32_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_gt_32_multipass) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_eq_32_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_eq_32_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(32)
        .channel_subtile(16)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, c_eq_32_multipass_multipixel) {
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        TEST_REQUIRES_X86_AVX512F;
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(32)
            .channel_subtile(16)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(163)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L32C16S1R__AVX512F_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(32)
          .channel_subtile(16)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(592)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_ARM_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMSIMD_X86_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, c_eq_1_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, c_eq_1_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, c_eq_1_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(1)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, c_gt_1_first_pass_plus_one) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, c_gt_1_first_pass_and_last_pass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, c_gt_1_multipass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, c_eq_1_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, c_eq_1_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, c_eq_1_multipass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(1)
            .channel_subtile(1)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(7)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(48)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, c_eq_1_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, c_eq_1_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, c_eq_1_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(1)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, c_gt_1_first_pass_plus_one) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, c_gt_1_first_pass_and_last_pass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, c_gt_1_multipass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, c_eq_1_multipass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(1)
            .channel_subtile(1)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(7)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__WASM_ACC2, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(48)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, c_eq_1_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, c_eq_1_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, c_eq_1_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(1)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, c_gt_1_first_pass_plus_one) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, c_gt_1_first_pass_and_last_pass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, c_gt_1_multipass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, c_eq_1_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, c_eq_1_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, c_eq_1_multipass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(1)
            .channel_subtile(1)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(7)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(48)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, c_eq_1_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, c_eq_1_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, c_eq_1_multipass) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(1)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, c_gt_1_first_pass_plus_one) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(7)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, c_gt_1_first_pass_and_last_pass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(13)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, c_gt_1_multipass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(7)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(13)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, c_eq_1_multipass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        for (size_t step = 2; step <= 6; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(6)
            .middle_pass_tile(6)
            .last_pass_tile(7)
            .channel_tile(1)
            .channel_subtile(1)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(7)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__WASM_ACC2, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(48)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, c_eq_1_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, c_eq_1_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, c_eq_1_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(1)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, c_gt_1_first_pass_plus_one) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, c_gt_1_first_pass_and_last_pass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, c_gt_1_multipass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, c_eq_1_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, c_eq_1_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, c_eq_1_multipass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(1)
            .channel_subtile(1)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(7)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(48)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, c_eq_1_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, c_eq_1_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, c_eq_1_multipass) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(1)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, c_gt_1_first_pass_plus_one) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, c_gt_1_first_pass_and_last_pass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(17)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, c_gt_1_multipass) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(17)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, c_eq_1_multipass_multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        for (size_t step = 2; step <= 8; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(8)
            .middle_pass_tile(8)
            .last_pass_tile(9)
            .channel_tile(1)
            .channel_subtile(1)
            .channel_round(1)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(7)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__WASM_ACC2, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(48)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(3)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(4)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, c_eq_1_multipass) {
  for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      for (size_t step = 2; step <= 2; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(3)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(4)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_multipass) {
  for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      for (size_t step = 2; step <= 2; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L1C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_eq_4_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(4)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(3)
    .channels(4)
    .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_eq_4_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(4)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(4)
    .channels(4)
    .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_eq_4_multipass) {
  for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_div_4_first_pass_plus_one) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_div_4_first_pass_and_last_pass) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_div_4_multipass) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_div_4_with_qmin) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_div_4_with_qmax) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_gt_4_first_pass_plus_one) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_gt_4_first_pass_and_last_pass) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_gt_4_multipass) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_eq_4_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_eq_4_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, c_eq_4_multipass_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      for (size_t step = 2; step <= 2; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(4)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(3)
    .channels(4)
    .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(4)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(4)
    .channels(4)
    .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_multipass) {
  for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_div_4_first_pass_plus_one) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_div_4_first_pass_and_last_pass) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_div_4_multipass) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_div_4_with_qmin) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_div_4_with_qmax) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_gt_4_first_pass_plus_one) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_gt_4_first_pass_and_last_pass) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_gt_4_multipass) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_multipass_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      for (size_t step = 2; step <= 2; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_2F2M2L4C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(5)
    .middle_pass_tile(5)
    .last_pass_tile(5)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(6)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(5)
    .middle_pass_tile(5)
    .last_pass_tile(5)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(10)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, c_eq_1_multipass) {
  for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      for (size_t step = 2; step <= 5; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(5)
    .middle_pass_tile(5)
    .last_pass_tile(5)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(6)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(5)
    .middle_pass_tile(5)
    .last_pass_tile(5)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(10)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_multipass) {
  for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      for (size_t step = 2; step <= 5; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_5F5M5L1C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(6)
    .middle_pass_tile(6)
    .last_pass_tile(7)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(7)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(6)
    .middle_pass_tile(6)
    .last_pass_tile(7)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(13)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, c_eq_1_multipass) {
  for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      for (size_t step = 2; step <= 6; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(6)
    .middle_pass_tile(6)
    .last_pass_tile(7)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(7)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(6)
    .middle_pass_tile(6)
    .last_pass_tile(7)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(13)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_multipass) {
  for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      for (size_t step = 2; step <= 6; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_6F6M7L1C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(8)
    .middle_pass_tile(8)
    .last_pass_tile(9)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(9)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(8)
    .middle_pass_tile(8)
    .last_pass_tile(9)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(17)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, c_eq_1_multipass) {
  for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      for (size_t step = 2; step <= 8; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(8)
    .middle_pass_tile(8)
    .last_pass_tile(9)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(9)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(8)
    .middle_pass_tile(8)
    .last_pass_tile(9)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(17)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_multipass) {
  for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      for (size_t step = 2; step <= 8; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_8F8M9L1C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}