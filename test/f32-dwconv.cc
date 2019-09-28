// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-dwconv.yaml
//   Generator: tools/generate-dwconv-test.py


#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/isa-checks.h>

#include "dwconv-microkernel-tester.h"


#if CPUINFO_ARCH_ARM64
  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
      }
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma);
    }
  }
#endif  // CPUINFO_ARCH_ARM64


#if CPUINFO_ARCH_ARM64
  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 12; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 12; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 12; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 12; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 12; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 9; channels < 12; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
      }
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }

  TEST(F32_DWCONV_UP4X9__AARCH64_NEONFMA_CORTEX_A55, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__aarch64_neonfma_cortex_a55);
    }
  }
#endif  // CPUINFO_ARCH_ARM64


#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  TEST(F32_DWCONV_UP4X9__NEONFMA, c_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, c_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, c_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, c_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, multipixel) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
      }
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEONFMA, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neonfma);
    }
  }
#endif  // CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64


#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  TEST(F32_DWCONV_UP4X9__NEON, c_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
  }

  TEST(F32_DWCONV_UP4X9__NEON, c_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, c_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, c_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, c_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, c_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, c_gt_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, c_gt_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, multipixel) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
      }
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }

  TEST(F32_DWCONV_UP4X9__NEON, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__neon);
    }
  }
#endif  // CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64


#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(F32_DWCONV_UP4X25__SSE, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
  }

  TEST(F32_DWCONV_UP4X25__SSE, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, c_gt_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, c_gt_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
      }
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }

  TEST(F32_DWCONV_UP4X25__SSE, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__sse);
    }
  }
#endif  // CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64


#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(F32_DWCONV_UP4X9__SSE, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
  }

  TEST(F32_DWCONV_UP4X9__SSE, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, c_gt_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, c_gt_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
      }
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }

  TEST(F32_DWCONV_UP4X9__SSE, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__sse);
    }
  }
#endif  // CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64


#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(F32_DWCONV_UP4X4__SSE, c_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
  }

  TEST(F32_DWCONV_UP4X4__SSE, c_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, c_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, c_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, c_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, c_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, c_gt_4_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, c_gt_4_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, multipixel) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, multipixel_with_step) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
      }
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, multipixel_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }

  TEST(F32_DWCONV_UP4X4__SSE, multipixel_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__sse);
    }
  }
#endif  // CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64


#if !CPUINFO_ARCH_ASMJS && !CPUINFO_ARCH_WASM
  TEST(F32_DWCONV_UP4X25__PSIMD, c_eq_4) {
    TEST_REQUIRES_PSIMD;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(25)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, c_div_4) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, c_div_4_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, c_div_4_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, c_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, c_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, c_gt_4_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, c_gt_4_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, multipixel) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, multipixel_with_step) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, multipixel_with_output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, multipixel_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X25__PSIMD, multipixel_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x25__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !CPUINFO_ARCH_ASMJS && !CPUINFO_ARCH_WASM


#if !CPUINFO_ARCH_ASMJS && !CPUINFO_ARCH_WASM
  TEST(F32_DWCONV_UP4X9__PSIMD, c_eq_4) {
    TEST_REQUIRES_PSIMD;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(9)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, c_div_4) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, c_div_4_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, c_div_4_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, c_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, c_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, c_gt_4_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, c_gt_4_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, multipixel) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, multipixel_with_step) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, multipixel_with_output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, multipixel_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X9__PSIMD, multipixel_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x9__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !CPUINFO_ARCH_ASMJS && !CPUINFO_ARCH_WASM


#if !CPUINFO_ARCH_ASMJS && !CPUINFO_ARCH_WASM
  TEST(F32_DWCONV_UP4X4__PSIMD, c_eq_4) {
    TEST_REQUIRES_PSIMD;
    DWConvMicrokernelTester()
      .cr(4)
      .kr(4)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, c_div_4) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, c_div_4_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, c_div_4_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, c_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, c_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, c_gt_4_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, c_gt_4_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, multipixel) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, multipixel_with_step) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(4)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, multipixel_with_output_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(4)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, multipixel_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_UP4X4__PSIMD, multipixel_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .cr(4)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f32_dwconv_ukernel_up4x4__psimd, DWConvMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !CPUINFO_ARCH_ASMJS && !CPUINFO_ARCH_WASM


TEST(F32_DWCONV_UP1X4__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(4)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_up1x4__scalar, DWConvMicrokernelTester::Variant::Scalar);
}

TEST(F32_DWCONV_UP1X4__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up1x4__scalar, DWConvMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}


TEST(F32_DWCONV_UP1X9__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(9)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
}

TEST(F32_DWCONV_UP1X9__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}


TEST(F32_DWCONV_UP1X25__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(25)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_up1x25__scalar, DWConvMicrokernelTester::Variant::Scalar);
}

TEST(F32_DWCONV_UP1X25__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR, c_gt_1_with_qmin) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .qmin(128)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR, c_gt_1_with_qmax) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .qmax(128)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up1x25__scalar, DWConvMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR, multipixel_with_qmin) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmin(128)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR, multipixel_with_qmax) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .qmax(128)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar, DWConvMicrokernelTester::Variant::Scalar);
  }
}
