// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-dwconv-unipass.yaml
//   Generator: tools/generate-dwconv-unipass-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_3P4C__WASMSIMD, c_eq_4) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(3)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
  }

  TEST(F32_DWCONV_3P4C__WASMSIMD, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMSIMD, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMSIMD, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMSIMD, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_3P4C__WASMSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMSIMD, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMSIMD, zero) {
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(3)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_3p4c__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_3P8C__WASMSIMD, c_eq_8) {
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(3)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
  }

  TEST(F32_DWCONV_3P8C__WASMSIMD, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMSIMD, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMSIMD, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMSIMD, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_3P8C__WASMSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMSIMD, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMSIMD, zero) {
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(3)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_3p8c__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_4P4C__WASMSIMD, c_eq_4) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(4)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
  }

  TEST(F32_DWCONV_4P4C__WASMSIMD, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMSIMD, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMSIMD, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMSIMD, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_4P4C__WASMSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMSIMD, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMSIMD, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_4p4c__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_4P8C__WASMSIMD, c_eq_8) {
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(4)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
  }

  TEST(F32_DWCONV_4P8C__WASMSIMD, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMSIMD, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMSIMD, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMSIMD, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_4P8C__WASMSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMSIMD, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMSIMD, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_4p8c__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_9P4C__WASMSIMD, c_eq_4) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_9P4C__WASMSIMD_ACC2, c_eq_4) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD_ACC2, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD_ACC2, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD_ACC2, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
      }
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMSIMD_ACC2, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_9P8C__WASMSIMD, c_eq_8) {
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(9)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_9P8C__WASMSIMD_ACC2, c_eq_8) {
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(9)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD_ACC2, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD_ACC2, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD_ACC2, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
      }
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD_ACC2, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMSIMD_ACC2, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_25P4C__WASMSIMD, c_eq_4) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(25)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
  }

  TEST(F32_DWCONV_25P4C__WASMSIMD, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMSIMD, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMSIMD, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMSIMD, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_25P4C__WASMSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMSIMD, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMSIMD, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_25p4c__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_25P8C__WASMSIMD, c_eq_8) {
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(25)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
  }

  TEST(F32_DWCONV_25P8C__WASMSIMD, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMSIMD, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMSIMD, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMSIMD, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_25P8C__WASMSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMSIMD, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMSIMD, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_25p8c__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, c_eq_4) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(3)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(3)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P4C__WASMRELAXEDSIMD_FMA, zero) {
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(3)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_3p4c__wasmrelaxedsimd_fma);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, c_eq_8) {
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(3)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 3; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(3)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(3)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_3P8C__WASMRELAXEDSIMD_FMA, zero) {
    for (uint32_t mz = 0; mz < 3; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(3)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_3p8c__wasmrelaxedsimd_fma);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, c_eq_4) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(4)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(4)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P4C__WASMRELAXEDSIMD_FMA, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(4)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_4p4c__wasmrelaxedsimd_fma);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, c_eq_8) {
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(4)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(4)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_4P8C__WASMRELAXEDSIMD_FMA, zero) {
    for (uint32_t mz = 0; mz < 4; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(4)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_4p8c__wasmrelaxedsimd_fma);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, c_eq_4) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(9)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P4C__WASMRELAXEDSIMD_FMA, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_9p4c__wasmrelaxedsimd_fma);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, c_eq_8) {
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(9)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_9P8C__WASMRELAXEDSIMD_FMA, zero) {
    for (uint32_t mz = 0; mz < 9; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(9)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_9p8c__wasmrelaxedsimd_fma);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, c_eq_4) {
    DWConvMicrokernelTester()
      .channel_tile(4)
      .kernel_tile(25)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, c_div_4) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, c_lt_4) {
    for (uint32_t channels = 1; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, c_gt_4) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .channel_tile(4)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P4C__WASMRELAXEDSIMD_FMA, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 8; channels < 64; channels += 12) {
        DWConvMicrokernelTester()
          .channel_tile(4)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(112)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_25p4c__wasmrelaxedsimd_fma);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, c_eq_8) {
    DWConvMicrokernelTester()
      .channel_tile(8)
      .kernel_tile(25)
      .channels(8)
      .Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, c_div_8) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, c_lt_8) {
    for (uint32_t channels = 1; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, c_gt_8) {
    for (uint32_t channels = 9; channels < 16; channels++) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, multipixel) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 40; channels += 7) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .width(5)
        .output_stride(43)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 16; channels < 128; channels += 24) {
      DWConvMicrokernelTester()
        .channel_tile(8)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(176)
        .Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_25P8C__WASMRELAXEDSIMD_FMA, zero) {
    for (uint32_t mz = 0; mz < 25; mz++) {
      for (uint32_t channels = 16; channels < 128; channels += 24) {
        DWConvMicrokernelTester()
          .channel_tile(8)
          .kernel_tile(25)
          .channels(channels)
          .input_offset(176)
          .zero_index(mz)
          .Test(xnn_f32_dwconv_ukernel_25p8c__wasmrelaxedsimd_fma);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_DWCONV_3P1C__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .channel_tile(1)
    .kernel_tile(3)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_3p1c__scalar);
}

TEST(F32_DWCONV_3P1C__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_3p1c__scalar);
  }
}

TEST(F32_DWCONV_3P1C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_3p1c__scalar);
  }
}

TEST(F32_DWCONV_3P1C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 3; step++) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(3)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_3p1c__scalar);
    }
  }
}

TEST(F32_DWCONV_3P1C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(3)
      .channels(channels)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_3p1c__scalar);
  }
}

TEST(F32_DWCONV_3P1C__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(3)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_ukernel_3p1c__scalar);
  }
}

TEST(F32_DWCONV_3P1C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 3; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(3)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_3p1c__scalar);
    }
  }
}

TEST(F32_DWCONV_3P1C__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .channel_tile(1)
    .kernel_tile(3)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_3p1c__scalar_acc2);
}

TEST(F32_DWCONV_3P1C__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_3p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P1C__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_3p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P1C__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 3; step++) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(3)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_3p1c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_3P1C__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(3)
      .channels(channels)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_3p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P1C__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(3)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_ukernel_3p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P1C__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 3; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(3)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_3p1c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_3P2C__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .channel_tile(2)
    .kernel_tile(3)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
}

TEST(F32_DWCONV_3P2C__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 3; step++) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(3)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
    }
  }
}

TEST(F32_DWCONV_3P2C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 3; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(3)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_3p2c__scalar);
    }
  }
}

TEST(F32_DWCONV_3P2C__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .channel_tile(2)
    .kernel_tile(3)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
}

TEST(F32_DWCONV_3P2C__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 3; step++) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(3)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_3P2C__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(3)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_3P2C__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 3; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(3)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_3p2c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_4P1C__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .channel_tile(1)
    .kernel_tile(4)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_4p1c__scalar);
}

TEST(F32_DWCONV_4P1C__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_4p1c__scalar);
  }
}

TEST(F32_DWCONV_4P1C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_4p1c__scalar);
  }
}

TEST(F32_DWCONV_4P1C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_4p1c__scalar);
    }
  }
}

TEST(F32_DWCONV_4P1C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(4)
      .channels(channels)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_4p1c__scalar);
  }
}

TEST(F32_DWCONV_4P1C__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(4)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_ukernel_4p1c__scalar);
  }
}

TEST(F32_DWCONV_4P1C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 4; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(4)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_4p1c__scalar);
    }
  }
}

TEST(F32_DWCONV_4P1C__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .channel_tile(1)
    .kernel_tile(4)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_4p1c__scalar_acc2);
}

TEST(F32_DWCONV_4P1C__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_4p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P1C__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_4p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P1C__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_4p1c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_4P1C__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(4)
      .channels(channels)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_4p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P1C__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(4)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_ukernel_4p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P1C__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 4; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(4)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_4p1c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_4P2C__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .channel_tile(2)
    .kernel_tile(4)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
}

TEST(F32_DWCONV_4P2C__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
    }
  }
}

TEST(F32_DWCONV_4P2C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 4; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(4)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_4p2c__scalar);
    }
  }
}

TEST(F32_DWCONV_4P2C__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .channel_tile(2)
    .kernel_tile(4)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
}

TEST(F32_DWCONV_4P2C__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_4P2C__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(4)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_4P2C__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 4; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(4)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_4p2c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_9P1C__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .channel_tile(1)
    .kernel_tile(9)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_9p1c__scalar);
}

TEST(F32_DWCONV_9P1C__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_9p1c__scalar);
  }
}

TEST(F32_DWCONV_9P1C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_9p1c__scalar);
  }
}

TEST(F32_DWCONV_9P1C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_9p1c__scalar);
    }
  }
}

TEST(F32_DWCONV_9P1C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_9p1c__scalar);
  }
}

TEST(F32_DWCONV_9P1C__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_ukernel_9p1c__scalar);
  }
}

TEST(F32_DWCONV_9P1C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_9p1c__scalar);
    }
  }
}

TEST(F32_DWCONV_9P1C__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .channel_tile(1)
    .kernel_tile(9)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_9p1c__scalar_acc2);
}

TEST(F32_DWCONV_9P1C__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_9p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P1C__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_9p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P1C__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_9p1c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_9P1C__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_9p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P1C__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(9)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_ukernel_9p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P1C__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_9p1c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_9P2C__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .channel_tile(2)
    .kernel_tile(9)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
}

TEST(F32_DWCONV_9P2C__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
    }
  }
}

TEST(F32_DWCONV_9P2C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_9p2c__scalar);
    }
  }
}

TEST(F32_DWCONV_9P2C__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .channel_tile(2)
    .kernel_tile(9)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
}

TEST(F32_DWCONV_9P2C__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_9P2C__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(9)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_9P2C__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 9; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(9)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_9p2c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_25P1C__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .channel_tile(1)
    .kernel_tile(25)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_25p1c__scalar);
}

TEST(F32_DWCONV_25P1C__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_25p1c__scalar);
  }
}

TEST(F32_DWCONV_25P1C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_25p1c__scalar);
  }
}

TEST(F32_DWCONV_25P1C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_25p1c__scalar);
    }
  }
}

TEST(F32_DWCONV_25P1C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(25)
      .channels(channels)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_25p1c__scalar);
  }
}

TEST(F32_DWCONV_25P1C__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(25)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_ukernel_25p1c__scalar);
  }
}

TEST(F32_DWCONV_25P1C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 25; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_25p1c__scalar);
    }
  }
}

TEST(F32_DWCONV_25P1C__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .channel_tile(1)
    .kernel_tile(25)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_25p1c__scalar_acc2);
}

TEST(F32_DWCONV_25P1C__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_25p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P1C__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_25p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P1C__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_25p1c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_25P1C__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(25)
      .channels(channels)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_25p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P1C__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    DWConvMicrokernelTester()
      .channel_tile(1)
      .kernel_tile(25)
      .channels(channels)
      .input_offset(48)
      .Test(xnn_f32_dwconv_ukernel_25p1c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P1C__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 25; mz++) {
    for (uint32_t channels = 2; channels < 16; channels += 3) {
      DWConvMicrokernelTester()
        .channel_tile(1)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(48)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_25p1c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_25P2C__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .channel_tile(2)
    .kernel_tile(25)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
}

TEST(F32_DWCONV_25P2C__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
    }
  }
}

TEST(F32_DWCONV_25P2C__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR, zero) {
  for (uint32_t mz = 0; mz < 25; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_25p2c__scalar);
    }
  }
}

TEST(F32_DWCONV_25P2C__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .channel_tile(2)
    .kernel_tile(25)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
}

TEST(F32_DWCONV_25P2C__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_25P2C__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .channel_tile(2)
      .kernel_tile(25)
      .channels(channels)
      .input_offset(80)
      .Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
  }
}

TEST(F32_DWCONV_25P2C__SCALAR_ACC2, zero) {
  for (uint32_t mz = 0; mz < 25; mz++) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .channel_tile(2)
        .kernel_tile(25)
        .channels(channels)
        .input_offset(80)
        .zero_index(mz)
        .Test(xnn_f32_dwconv_ukernel_25p2c__scalar_acc2);
    }
  }
}