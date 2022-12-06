// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-dwconv-multipass-minmax.yaml
//   Generator: tools/generate-dwconv-multipass-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(3)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_eq_4_multipass) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        for (size_t step = 2; step <= 2; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(2)
            .middle_pass_tile(2)
            .last_pass_tile(2)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_eq_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(3)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_eq_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_div_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_div_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_gt_4_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_gt_4_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        for (size_t step = 2; step <= 2; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(2)
            .middle_pass_tile(2)
            .last_pass_tile(2)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L4C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(3)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_eq_8_multipass) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_div_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_gt_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        for (size_t step = 2; step <= 2; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(2)
            .middle_pass_tile(2)
            .last_pass_tile(2)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_eq_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(3)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_eq_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(8)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_eq_8_multipass) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(8)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_div_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_div_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_div_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_gt_8_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_gt_8_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_gt_8_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_eq_8_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_eq_8_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(8)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, c_eq_8_multipass_multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        for (size_t step = 2; step <= 2; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(2)
            .middle_pass_tile(2)
            .last_pass_tile(2)
            .channel_tile(8)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(43)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L8C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(8)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(176)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(3)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(4)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_eq_16_multipass) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_div_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_gt_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        for (size_t step = 2; step <= 2; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(2)
            .middle_pass_tile(2)
            .last_pass_tile(2)
            .channel_tile(16)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_eq_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(3)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_eq_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(16)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(4)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_eq_16_multipass) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      TEST_REQUIRES_X86_SSE;
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(16)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_div_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_div_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_div_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_gt_16_first_pass_plus_one) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_gt_16_first_pass_and_last_pass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_gt_16_multipass) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 17; channels < 32; channels++) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_eq_16_first_pass_plus_one_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_eq_16_first_pass_and_last_pass_multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(16)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, c_eq_16_multipass_multipixel) {
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        TEST_REQUIRES_X86_SSE;
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        for (size_t step = 2; step <= 2; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(2)
            .middle_pass_tile(2)
            .last_pass_tile(2)
            .channel_tile(16)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
        }
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(83)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_2F2M2L16C4S4R__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(16)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(304)
          .Test(xnn_f32_dwconv_minmax_ukernel_2f2m2l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
