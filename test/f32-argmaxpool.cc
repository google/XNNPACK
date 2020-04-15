// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-argmaxpool.yaml
//   Generator: tools/generate-argmaxpool-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/argmaxpool.h>
#include "argmaxpool-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_eq_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(4)
      .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_eq_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_eq_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(4)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_eq_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(4)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_eq_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(4)
        .channels(4)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_eq_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(4)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_div_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_div_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_div_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_div_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_div_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_div_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_lt_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_lt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_lt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_lt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_lt_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_lt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_gt_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_gt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_gt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_gt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_gt_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, channels_gt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, few_output_pixels) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, few_output_pixels_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .qmin(192)
            .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, few_output_pixels_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .qmax(192)
            .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__SSE2_C4, few_output_pixels_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            ArgMaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(4)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_argmaxpool_ukernel_4x__sse2_c4);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_eq_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(4)
      .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_eq_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_eq_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(4)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_eq_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(4)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_eq_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(4)
        .channels(4)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_eq_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(4)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_div_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_div_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_div_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_div_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_div_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_div_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_lt_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_lt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_lt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_lt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_lt_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_lt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_gt_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_gt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_gt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_gt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(4)
        .pooling_tile(4)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_gt_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, channels_gt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, few_output_pixels) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, few_output_pixels_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .qmin(192)
            .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, few_output_pixels_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .qmax(192)
            .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_4X__PSIMD_C4, few_output_pixels_with_step) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            ArgMaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(4)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_argmaxpool_ukernel_4x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_eq_1_unipass_fulltile) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(4)
    .pooling_tile(4)
    .channels(1)
    .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_input_offset) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(4)
    .pooling_tile(4)
    .channels(1)
    .input_offset(3)
    .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_qmin) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(4)
    .pooling_tile(4)
    .channels(1)
    .qmin(192)
    .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_qmax) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(4)
    .pooling_tile(4)
    .channels(1)
    .qmax(192)
    .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_eq_1_unipass_subtile) {
  for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(4)
      .channels(1)
      .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_eq_1_unipass_subtile_with_input_offset) {
  for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(4)
      .channels(1)
      .input_offset(3)
      .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_gt_1_unipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(channels)
      .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_input_offset) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(channels)
      .input_offset(3)
      .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(channels)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(4)
      .pooling_tile(4)
      .channels(channels)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_gt_1_unipass_subtile) {
  for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(4)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, channels_gt_1_unipass_subtile_with_input_offset) {
  for (size_t pooling_elements = 2; pooling_elements < 4; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(4)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, few_output_pixels) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, few_output_pixels_with_input_offset) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .input_offset(7)
          .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, few_output_pixels_with_qmin) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .qmin(192)
          .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, few_output_pixels_with_qmax) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .qmax(192)
          .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, few_output_pixels_with_output_stride) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(4)
          .channels(channels)
          .output_stride(7)
          .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_4X__SCALAR_C1, few_output_pixels_with_step) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 4; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (size_t step = 2; step <= pooling_elements; step++) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(4)
            .step(step)
            .channels(channels)
            .output_stride(7)
            .Test(xnn_f32_argmaxpool_ukernel_4x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_eq_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(4)
      .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_eq_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_eq_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(4)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_eq_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(4)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_eq_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(4)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_eq_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_div_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_div_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_div_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_div_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_div_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_div_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_lt_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_lt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_lt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_lt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_lt_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_lt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_gt_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_gt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_gt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_gt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_gt_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, channels_gt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, few_output_pixels) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, few_output_pixels_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .qmin(192)
            .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, few_output_pixels_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .qmax(192)
            .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__SSE2_C4, few_output_pixels_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            ArgMaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_argmaxpool_ukernel_9x__sse2_c4);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_eq_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(4)
      .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_eq_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_eq_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(4)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_eq_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(4)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_eq_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(4)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_eq_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_div_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_div_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(37)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_div_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_div_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_div_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_div_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_lt_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_lt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_lt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_lt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_lt_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_lt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_gt_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_gt_4_unipass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_gt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_gt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(9)
        .pooling_tile(9)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_gt_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, channels_gt_4_unipass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, few_output_pixels) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, few_output_pixels_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .qmin(192)
            .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, few_output_pixels_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .qmax(192)
            .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9X__PSIMD_C4, few_output_pixels_with_step) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            ArgMaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_argmaxpool_ukernel_9x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_eq_1_unipass_fulltile) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9)
    .channels(1)
    .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_input_offset) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9)
    .channels(1)
    .input_offset(3)
    .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_qmin) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9)
    .channels(1)
    .qmin(192)
    .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_eq_1_unipass_fulltile_with_qmax) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(9)
    .pooling_tile(9)
    .channels(1)
    .qmax(192)
    .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_eq_1_unipass_subtile) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9)
      .channels(1)
      .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_eq_1_unipass_subtile_with_input_offset) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9)
      .channels(1)
      .input_offset(3)
      .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_gt_1_unipass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channels)
      .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_input_offset) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channels)
      .input_offset(3)
      .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channels)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_gt_1_unipass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(9)
      .pooling_tile(9)
      .channels(channels)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_gt_1_unipass_subtile) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, channels_gt_1_unipass_subtile_with_input_offset) {
  for (size_t pooling_elements = 2; pooling_elements < 9; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, few_output_pixels) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, few_output_pixels_with_input_offset) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .input_offset(7)
          .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, few_output_pixels_with_qmin) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .qmin(192)
          .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, few_output_pixels_with_qmax) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .qmax(192)
          .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, few_output_pixels_with_output_stride) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9)
          .channels(channels)
          .output_stride(7)
          .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9X__SCALAR_C1, few_output_pixels_with_step) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 2; pooling_elements <= 9; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (size_t step = 2; step <= pooling_elements; step++) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9)
            .step(step)
            .channels(channels)
            .output_stride(7)
            .Test(xnn_f32_argmaxpool_ukernel_9x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(23)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(4)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_eq_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(4)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_div_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(4)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_lt_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_multipass_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, channels_gt_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, few_output_pixels) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, few_output_pixels_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(192)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, few_output_pixels_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(192)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__SSE2_C4, few_output_pixels_with_step) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            ArgMaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_argmaxpool_ukernel_9p8x__sse2_c4);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .input_offset(7)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(4)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(pooling_elements)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(23)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 8; channels < 32; channels += 4) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(5)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels < 4; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(5)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_twopass_fulltile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(11)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 5; channels < 8; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_twopass_subtile_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_multipass) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(4)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_multipass_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(4)
        .input_offset(7)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_multipass_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(4)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_eq_4_multipass_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(4)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_multipass) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_multipass_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(37)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_multipass_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_div_4_multipass_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 8; channels < 32; channels += 4) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_multipass) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_multipass_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(4)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_multipass_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_lt_4_multipass_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 1; channels < 4; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_multipass) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_multipass_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(11)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_multipass_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, channels_gt_4_multipass_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
      for (size_t channels = 5; channels < 8; channels++) {
        ArgMaxPoolMicrokernelTester()
          .pooling_elements(17)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, few_output_pixels) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, few_output_pixels_with_input_offset) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .input_offset(23)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, few_output_pixels_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmin(192)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, few_output_pixels_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .qmax(192)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, few_output_pixels_with_output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .channels(channels)
            .output_stride(23)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_ARGMAXPOOL_9P8X__PSIMD_C4, few_output_pixels_with_step) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
        for (size_t channels = 1; channels <= 20; channels += 3) {
          for (size_t step = 2; step <= pooling_elements; step++) {
            ArgMaxPoolMicrokernelTester()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(9, 8)
              .step(step)
              .channels(channels)
              .output_stride(23)
              .Test(xnn_f32_argmaxpool_ukernel_9p8x__psimd_c4, ArgMaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC



TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile_with_input_offset) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .input_offset(3)
    .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile_with_qmin) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .qmin(192)
    .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_twopass_fulltile_with_qmax) {
  ArgMaxPoolMicrokernelTester()
    .pooling_elements(17)
    .pooling_tile(9, 8)
    .channels(1)
    .qmax(192)
    .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_twopass_subtile) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_twopass_subtile_with_input_offset) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(pooling_elements)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile_with_input_offset) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .input_offset(3)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile_with_qmin) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_twopass_fulltile_with_qmax) {
  for (size_t channels = 2; channels < 10; channels++) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(channels)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_twopass_subtile) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_twopass_subtile_with_input_offset) {
  for (size_t pooling_elements = 10; pooling_elements < 17; pooling_elements++) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_multipass) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(1)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_multipass_with_input_offset) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(1)
      .input_offset(3)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_multipass_with_qmin) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(1)
      .qmin(192)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_eq_1_multipass_with_qmax) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    ArgMaxPoolMicrokernelTester()
      .pooling_elements(17)
      .pooling_tile(9, 8)
      .channels(1)
      .qmax(192)
      .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_multipass) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_multipass_with_input_offset) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .input_offset(3)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_multipass_with_qmin) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmin(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, channels_gt_1_multipass_with_qmax) {
  for (size_t pooling_elements = 18; pooling_elements <= 33; pooling_elements += 3) {
    for (size_t channels = 2; channels < 10; channels++) {
      ArgMaxPoolMicrokernelTester()
        .pooling_elements(17)
        .pooling_tile(9, 8)
        .channels(channels)
        .qmax(192)
        .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, few_output_pixels) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, few_output_pixels_with_input_offset) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .input_offset(7)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, few_output_pixels_with_qmin) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmin(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, few_output_pixels_with_qmax) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .qmax(192)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, few_output_pixels_with_output_stride) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        ArgMaxPoolMicrokernelTester()
          .output_pixels(output_pixels)
          .pooling_elements(pooling_elements)
          .pooling_tile(9, 8)
          .channels(channels)
          .output_stride(7)
          .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_ARGMAXPOOL_9P8X__SCALAR_C1, few_output_pixels_with_step) {
  for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
    for (size_t pooling_elements = 10; pooling_elements <= 17; pooling_elements++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        for (size_t step = 2; step <= pooling_elements; step++) {
          ArgMaxPoolMicrokernelTester()
            .output_pixels(output_pixels)
            .pooling_elements(pooling_elements)
            .pooling_tile(9, 8)
            .step(step)
            .channels(channels)
            .output_stride(7)
            .Test(xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1, ArgMaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}