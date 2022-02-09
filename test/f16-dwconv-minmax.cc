// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-dwconv-minmax.yaml
//   Generator: tools/generate-dwconv-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, c_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, c_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, c_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, c_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, c_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, c_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, c_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, c_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, c_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(25)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, c_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, c_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, c_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, c_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(25)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, c_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, c_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, c_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, c_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, c_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, c_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, c_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, c_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, c_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, c_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, c_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, c_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, c_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, c_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, c_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, c_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, c_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, c_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, c_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, c_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, c_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, c_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, c_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, c_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, c_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, c_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, c_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, c_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(4)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, c_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, c_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, c_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, c_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(4)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, c_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, c_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, c_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(3)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(3)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, c_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(3)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, c_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, c_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, c_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(3)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, c_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(3)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, c_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, c_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, c_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(3)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, c_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(3)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, c_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, c_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, c_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(3)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, c_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(3)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, c_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, c_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, c_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(3)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, c_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(3)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, c_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, c_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, c_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, multipixel) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, multipixel_with_step) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, input_offset) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__NEONFP16ARITH_ACC2, zero) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(3)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(25)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X25__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(25)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(25)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X25__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(25)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, c_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(25)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, c_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, c_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, c_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, c_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(25)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, c_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, c_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, c_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(25)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X25__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(25)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(9)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X9__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(9)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(9)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X9__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(9)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, c_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, c_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, c_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, c_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, c_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(9)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, c_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, c_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, c_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(9)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X9__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(9)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(4)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X4__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(4)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(4)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X4__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(4)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, c_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(4)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, c_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, c_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, c_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, c_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(4)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, c_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, c_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, c_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(4)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X4__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(4)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(3)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(3)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, c_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(8)
      .kr(3)
      .channels(8)
      .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, c_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, c_div_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, c_div_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, c_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, c_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, c_gt_8_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, c_gt_8_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(8)
        .width(5)
        .output_stride(43)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .cr(8)
        .kr(3)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP8X3__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .cr(8)
          .kr(3)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(3)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(3)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, c_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(16)
      .kr(3)
      .channels(16)
      .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, c_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, c_div_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, c_div_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, c_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, c_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, c_gt_16_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, c_gt_16_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 17; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(16)
        .width(5)
        .output_stride(83)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 80; channels += 15) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 32; channels < 256; channels += 48) {
      DWConvMicrokernelTester()
        .cr(16)
        .kr(3)
        .channels(channels)
        .input_offset(304)
        .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP16X3__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 32; channels < 256; channels += 48) {
        DWConvMicrokernelTester()
          .cr(16)
          .kr(3)
          .channels(channels)
          .input_offset(304)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, c_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(3)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, c_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, c_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, c_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(3)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, c_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    DWConvMicrokernelTester()
      .cr(32)
      .kr(3)
      .channels(32)
      .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, c_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, c_div_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, c_div_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, c_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 1; channels < 32; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, c_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, c_gt_32_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, c_gt_32_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 33; channels < 64; channels++) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, multipixel) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, multipixel_with_step) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, multipixel_with_output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(32)
        .width(5)
        .output_stride(163)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, multipixel_with_qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmin(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, multipixel_with_qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels <= 160; channels += 31) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .width(3)
        .qmax(128)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, input_offset) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t channels = 64; channels < 512; channels += 96) {
      DWConvMicrokernelTester()
        .cr(32)
        .kr(3)
        .channels(channels)
        .input_offset(592)
        .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_DWCONV_MINMAX_UP32X3__FMA3_ACC2, zero) {
    TEST_REQUIRES_X86_FMA3;
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 64; channels < 512; channels += 96) {
        DWConvMicrokernelTester()
          .cr(32)
          .kr(3)
          .channels(channels)
          .input_offset(592)
          .zero_index(mz)
          .Test(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
