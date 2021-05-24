// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-dwconv-minmax.yaml
//   Generator: tools/generate-dwconv-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 12; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 12; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 12; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 12; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 12; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 12; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 12; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__AARCH64_NEONFMA_CORTEX_A55, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 12; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEONFMA_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEONFMA_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEONFMA_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEONFMA_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEONFMA_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEONFMA_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, c_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, c_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, c_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, c_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, c_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, c_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, c_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, c_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__NEON_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__NEON_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, c_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, c_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, c_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, c_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, c_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, c_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, c_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, c_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__NEON_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__NEON_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, c_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, c_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, c_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, c_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, c_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, c_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, c_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, c_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__NEON_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__NEON_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, c_gt_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, c_gt_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, c_gt_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, c_gt_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__SSE_ACC2, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, c_eq_8) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, c_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, c_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, c_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, c_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__SSE_ACC2, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, c_gt_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, c_gt_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, c_gt_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, c_gt_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__SSE_ACC2, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, c_eq_8) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, c_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, c_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, c_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, c_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__SSE_ACC2, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, c_gt_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, c_gt_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, c_gt_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, c_gt_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__SSE_ACC2, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, c_eq_8) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, c_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, c_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, c_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, c_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, input_offset) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__SSE_ACC2, zero) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2, xnn_init_f32_minmax_sse_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, c_eq_8) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, c_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, c_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, c_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, c_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__AVX_ACC2, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, c_eq_16) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, c_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, c_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, c_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, c_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX_ACC2, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, c_eq_8) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, c_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, c_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, c_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, c_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__AVX_ACC2, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, c_eq_16) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, c_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, c_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, c_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, c_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX_ACC2, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, c_eq_8) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, c_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, c_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, c_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, c_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__AVX_ACC2, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, c_eq_16) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, c_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, c_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, c_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_AVX;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, c_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX_ACC2, zero) {
    TEST_REQUIRES_X86_AVX;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f32_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, c_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, c_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, c_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, c_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, c_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X25__AVX512F_ACC2, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, c_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(25)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, c_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, c_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, c_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, c_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(25)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, c_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, c_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, c_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X25__AVX512F_ACC2, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, c_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, c_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, c_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, c_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, c_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X9__AVX512F_ACC2, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, c_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, c_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, c_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, c_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, c_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, c_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, c_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, c_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X9__AVX512F_ACC2, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, c_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, c_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, c_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, c_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, c_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP16X4__AVX512F_ACC2, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, c_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(4)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, c_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, c_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, c_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, c_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(4)
      .channels(32)
      .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, c_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, c_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, c_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, multipixel) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, input_offset) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP32X4__AVX512F_ACC2, zero) {
    TEST_REQUIRES_X86_AVX512F;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_ARM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_ARM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X25__WASMSIMD_X86_ACC2, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X25__WASMSIMD_X86_ACC2, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_ARM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_ARM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X9__WASMSIMD_X86_ACC2, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X9__WASMSIMD_X86_ACC2, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_ARM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_ARM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, c_eq_4) {
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, c_div_4_with_qmin) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, c_div_4_with_qmax) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, c_gt_4_with_qmin) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, c_gt_4_with_qmax) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP4X4__WASMSIMD_X86_ACC2, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, c_eq_8) {
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, c_div_8_with_qmin) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, c_div_8_with_qmax) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, c_gt_8_with_qmin) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, c_gt_8_with_qmax) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP8X4__WASMSIMD_X86_ACC2, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, c_gt_1_with_qmin) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, c_gt_1_with_qmax) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 2; channels < 16; channels += 3) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(4)
          .channels(channels)
          .input_offset(48)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, c_gt_1_with_qmin) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, c_gt_1_with_qmax) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X4__WASM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 2; channels < 16; channels += 3) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(4)
          .channels(channels)
          .input_offset(48)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(2)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, c_div_2_with_qmin) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, c_div_2_with_qmax) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, c_gt_2_with_qmin) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, c_gt_2_with_qmax) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, input_offset) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .input_offset(80)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 4; channels < 32; channels += 6) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(4)
          .channels(channels)
          .input_offset(80)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(2)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, c_div_2_with_qmin) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, c_div_2_with_qmax) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, c_gt_2_with_qmin) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, c_gt_2_with_qmax) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, input_offset) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .input_offset(80)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X4__WASM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 4; channels < 32; channels += 6) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(4)
          .channels(channels)
          .input_offset(80)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, c_gt_1_with_qmin) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, c_gt_1_with_qmax) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 2; channels < 16; channels += 3) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(9)
          .channels(channels)
          .input_offset(48)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, c_gt_1_with_qmin) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, c_gt_1_with_qmax) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X9__WASM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 2; channels < 16; channels += 3) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(9)
          .channels(channels)
          .input_offset(48)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(2)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, c_div_2_with_qmin) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, c_div_2_with_qmax) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, c_gt_2_with_qmin) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, c_gt_2_with_qmax) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, input_offset) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .input_offset(80)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 4; channels < 32; channels += 6) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(9)
          .channels(channels)
          .input_offset(80)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(2)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, c_div_2_with_qmin) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, c_div_2_with_qmax) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, c_gt_2_with_qmin) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, c_gt_2_with_qmax) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, input_offset) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .input_offset(80)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X9__WASM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 4; channels < 32; channels += 6) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(9)
          .channels(channels)
          .input_offset(80)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, c_gt_1_with_qmin) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, c_gt_1_with_qmax) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 2; channels < 16; channels += 3) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(25)
          .channels(channels)
          .input_offset(48)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(1)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, c_gt_1_with_qmin) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, c_gt_1_with_qmax) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, input_offset) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP1X25__WASM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 2; channels < 16; channels += 3) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(25)
          .channels(channels)
          .input_offset(48)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(2)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, c_div_2_with_qmin) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, c_div_2_with_qmax) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, c_gt_2_with_qmin) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, c_gt_2_with_qmax) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, input_offset) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .input_offset(80)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 4; channels < 32; channels += 6) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(25)
          .channels(channels)
          .input_offset(80)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(2)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, c_div_2_with_qmin) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, c_div_2_with_qmax) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, c_gt_2_with_qmin) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, c_gt_2_with_qmax) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, multipixel_with_qmin) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, multipixel_with_qmax) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, input_offset) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .input_offset(80)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_DWCONV_MINMAX_UP2X25__WASM_ACC2, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 4; channels < 32; channels += 6) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(25)
          .channels(channels)
          .input_offset(80)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2, xnn_init_f32_minmax_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(4)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 4; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(4)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X4__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 4; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(4)
    .channels(2)
    .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, c_div_2_with_qmin) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, c_div_2_with_qmax) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, c_gt_2_with_qmin) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, c_gt_2_with_qmax) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 4; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(4)
    .channels(2)
    .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, c_div_2_with_qmin) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, c_div_2_with_qmax) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, c_gt_2_with_qmin) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, c_gt_2_with_qmax) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X4__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 4; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(9)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(9)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X9__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(9)
    .channels(2)
    .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, c_div_2_with_qmin) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, c_div_2_with_qmax) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, c_gt_2_with_qmin) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, c_gt_2_with_qmax) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(9)
    .channels(2)
    .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, c_div_2_with_qmin) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, c_div_2_with_qmax) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, c_gt_2_with_qmin) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, c_gt_2_with_qmax) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X9__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(25)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 25; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(25)
    .channels(1)
    .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP1X25__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 25; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(25)
    .channels(2)
    .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, c_div_2_with_qmin) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, c_div_2_with_qmax) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, c_gt_2_with_qmin) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, c_gt_2_with_qmax) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 25; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(25)
    .channels(2)
    .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, c_div_2_with_qmin) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, c_div_2_with_qmax) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, c_gt_2_with_qmin) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, c_gt_2_with_qmax) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_DWCONV_MINMAX_UP2X25__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 25; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2, xnn_init_f32_minmax_scalar_params);
    }
  }
}