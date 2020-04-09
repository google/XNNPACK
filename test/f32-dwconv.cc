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


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP1X4__WASM, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_up1x4__wasm);
  }

  TEST(F32_DWCONV_UP1X4__WASM, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up1x4__wasm);
    }
  }

  TEST(F32_DWCONV_UP1X4__WASM, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up1x4__wasm);
    }
  }

  TEST(F32_DWCONV_UP1X4__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up1x4__wasm);
      }
    }
  }

  TEST(F32_DWCONV_UP1X4__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_up1x4__wasm);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP1X4__WASM_ACC2, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_up1x4__wasm_acc2);
  }

  TEST(F32_DWCONV_UP1X4__WASM_ACC2, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up1x4__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP1X4__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up1x4__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP1X4__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up1x4__wasm_acc2);
      }
    }
  }

  TEST(F32_DWCONV_UP1X4__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_up1x4__wasm_acc2);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP2X4__WASM, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(2)
      .Test(xnn_f32_dwconv_ukernel_up2x4__wasm);
  }

  TEST(F32_DWCONV_UP2X4__WASM, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up2x4__wasm);
      }
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP2X4__WASM_ACC2, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(2)
      .Test(xnn_f32_dwconv_ukernel_up2x4__wasm_acc2);
  }

  TEST(F32_DWCONV_UP2X4__WASM_ACC2, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM_ACC2, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM_ACC2, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 4; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(4)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up2x4__wasm_acc2);
      }
    }
  }

  TEST(F32_DWCONV_UP2X4__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_ukernel_up2x4__wasm_acc2);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP1X9__WASM, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_up1x9__wasm);
  }

  TEST(F32_DWCONV_UP1X9__WASM, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up1x9__wasm);
    }
  }

  TEST(F32_DWCONV_UP1X9__WASM, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up1x9__wasm);
    }
  }

  TEST(F32_DWCONV_UP1X9__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up1x9__wasm);
      }
    }
  }

  TEST(F32_DWCONV_UP1X9__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_up1x9__wasm);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP1X9__WASM_ACC2, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_up1x9__wasm_acc2);
  }

  TEST(F32_DWCONV_UP1X9__WASM_ACC2, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up1x9__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP1X9__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up1x9__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP1X9__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up1x9__wasm_acc2);
      }
    }
  }

  TEST(F32_DWCONV_UP1X9__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_up1x9__wasm_acc2);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP2X9__WASM, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(2)
      .Test(xnn_f32_dwconv_ukernel_up2x9__wasm);
  }

  TEST(F32_DWCONV_UP2X9__WASM, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up2x9__wasm);
      }
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP2X9__WASM_ACC2, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(2)
      .Test(xnn_f32_dwconv_ukernel_up2x9__wasm_acc2);
  }

  TEST(F32_DWCONV_UP2X9__WASM_ACC2, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM_ACC2, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM_ACC2, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 9; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(9)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up2x9__wasm_acc2);
      }
    }
  }

  TEST(F32_DWCONV_UP2X9__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_ukernel_up2x9__wasm_acc2);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP1X25__WASM, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_up1x25__wasm);
  }

  TEST(F32_DWCONV_UP1X25__WASM, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up1x25__wasm);
    }
  }

  TEST(F32_DWCONV_UP1X25__WASM, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up1x25__wasm);
    }
  }

  TEST(F32_DWCONV_UP1X25__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up1x25__wasm);
      }
    }
  }

  TEST(F32_DWCONV_UP1X25__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_up1x25__wasm);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP1X25__WASM_ACC2, c_eq_1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_up1x25__wasm_acc2);
  }

  TEST(F32_DWCONV_UP1X25__WASM_ACC2, c_gt_1) {
    for (uint32_t channels = 2; channels < 10; channels++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up1x25__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP1X25__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up1x25__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP1X25__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(1)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up1x25__wasm_acc2);
      }
    }
  }

  TEST(F32_DWCONV_UP1X25__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(1)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_up1x25__wasm_acc2);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP2X25__WASM, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(2)
      .Test(xnn_f32_dwconv_ukernel_up2x25__wasm);
  }

  TEST(F32_DWCONV_UP2X25__WASM, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm);
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up2x25__wasm);
      }
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm);
    }
  }
#endif  // XNN_ARCH_WASM


#if XNN_ARCH_WASM
  TEST(F32_DWCONV_UP2X25__WASM_ACC2, c_eq_2) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(2)
      .Test(xnn_f32_dwconv_ukernel_up2x25__wasm_acc2);
  }

  TEST(F32_DWCONV_UP2X25__WASM_ACC2, c_div_2) {
    for (uint32_t channels = 4; channels < 32; channels += 6) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM_ACC2, c_lt_2) {
    for (uint32_t channels = 1; channels < 2; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM_ACC2, c_gt_2) {
    for (uint32_t channels = 3; channels < 4; channels++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM_ACC2, multipixel) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm_acc2);
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      for (size_t step = 2; step <= 25; step++) {
        DWConvMicrokernelTester()
          .cr(2)
          .kr(25)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_up2x25__wasm_acc2);
      }
    }
  }

  TEST(F32_DWCONV_UP2X25__WASM_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(2)
        .width(5)
        .output_stride(13)
        .Test(xnn_f32_dwconv_ukernel_up2x25__wasm_acc2);
    }
  }
#endif  // XNN_ARCH_WASM


TEST(F32_DWCONV_UP1X4__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(4)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_up1x4__scalar);
}

TEST(F32_DWCONV_UP1X4__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar);
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar);
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
        .Test(xnn_f32_dwconv_ukernel_up1x4__scalar);
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
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar);
  }
}


TEST(F32_DWCONV_UP1X4__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(4)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_up1x4__scalar_acc2);
}

TEST(F32_DWCONV_UP1X4__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up1x4__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_UP1X4__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(4)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_up1x4__scalar_acc2);
  }
}


TEST(F32_DWCONV_UP2X4__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(4)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_up2x4__scalar);
}

TEST(F32_DWCONV_UP2X4__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar);
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar);
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar);
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar);
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up2x4__scalar);
    }
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar);
  }
}


TEST(F32_DWCONV_UP2X4__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(4)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_up2x4__scalar_acc2);
}

TEST(F32_DWCONV_UP2X4__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 4; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(4)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up2x4__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_UP2X4__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(4)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_up2x4__scalar_acc2);
  }
}


TEST(F32_DWCONV_UP1X9__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(9)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_up1x9__scalar);
}

TEST(F32_DWCONV_UP1X9__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar);
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar);
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
        .Test(xnn_f32_dwconv_ukernel_up1x9__scalar);
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
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar);
  }
}


TEST(F32_DWCONV_UP1X9__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(9)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_up1x9__scalar_acc2);
}

TEST(F32_DWCONV_UP1X9__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up1x9__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_UP1X9__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(9)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_up1x9__scalar_acc2);
  }
}


TEST(F32_DWCONV_UP2X9__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(9)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_up2x9__scalar);
}

TEST(F32_DWCONV_UP2X9__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar);
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar);
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar);
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar);
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up2x9__scalar);
    }
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar);
  }
}


TEST(F32_DWCONV_UP2X9__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(9)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_up2x9__scalar_acc2);
}

TEST(F32_DWCONV_UP2X9__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 9; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(9)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up2x9__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_UP2X9__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(9)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_up2x9__scalar_acc2);
  }
}


TEST(F32_DWCONV_UP1X25__SCALAR, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(25)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_up1x25__scalar);
}

TEST(F32_DWCONV_UP1X25__SCALAR, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar);
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar);
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
        .Test(xnn_f32_dwconv_ukernel_up1x25__scalar);
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
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar);
  }
}


TEST(F32_DWCONV_UP1X25__SCALAR_ACC2, c_eq_1) {
  DWConvMicrokernelTester()
    .cr(1)
    .kr(25)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_up1x25__scalar_acc2);
}

TEST(F32_DWCONV_UP1X25__SCALAR_ACC2, c_gt_1) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .cr(1)
        .kr(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up1x25__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_UP1X25__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .cr(1)
      .kr(25)
      .channels(1)
      .width(5)
      .output_stride(7)
      .Test(xnn_f32_dwconv_ukernel_up1x25__scalar_acc2);
  }
}


TEST(F32_DWCONV_UP2X25__SCALAR, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(25)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_up2x25__scalar);
}

TEST(F32_DWCONV_UP2X25__SCALAR, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar);
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar);
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar);
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar);
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up2x25__scalar);
    }
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar);
  }
}


TEST(F32_DWCONV_UP2X25__SCALAR_ACC2, c_eq_2) {
  DWConvMicrokernelTester()
    .cr(2)
    .kr(25)
    .channels(2)
    .Test(xnn_f32_dwconv_ukernel_up2x25__scalar_acc2);
}

TEST(F32_DWCONV_UP2X25__SCALAR_ACC2, c_div_2) {
  for (uint32_t channels = 4; channels < 32; channels += 6) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR_ACC2, c_lt_2) {
  for (uint32_t channels = 1; channels < 2; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR_ACC2, c_gt_2) {
  for (uint32_t channels = 3; channels < 4; channels++) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR_ACC2, multipixel) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar_acc2);
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    for (size_t step = 2; step <= 25; step++) {
      DWConvMicrokernelTester()
        .cr(2)
        .kr(25)
        .channels(channels)
        .width(3)
        .step(step)
        .Test(xnn_f32_dwconv_ukernel_up2x25__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_UP2X25__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 10; channels += 1) {
    DWConvMicrokernelTester()
      .cr(2)
      .kr(25)
      .channels(2)
      .width(5)
      .output_stride(13)
      .Test(xnn_f32_dwconv_ukernel_up2x25__scalar_acc2);
  }
}
