// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-chw-microkernel-tester.h"

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_CHW_3X3P1__SSE, input_width_eq_4) {
    TEST_REQUIRES_X86_SSE;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
  }

  TEST(F32_DWCONV_CHW_3X3P1__SSE, input_width_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__SSE, input_width_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__SSE, input_width_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__SSE, input_width_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__SSE, input_tuple_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__SSE, output_height_gt_1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t output_height = 2; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(1)
          .padding_right(1)
          .padding_top(1)
          .padding_bottom(1)
          .kernel_height(3)
          .kernel_width(3)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
      }
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__SSE, output_width_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__SSE, output_tuple_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__SSE, chw_layout) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(5)
        .output_width_stride(input_width)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__sse);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, input_width_eq_4) {
    TEST_REQUIRES_PSIMD;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, input_width_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, input_width_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, input_width_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, input_width_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, input_tuple_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, output_height_gt_1) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_height = 2; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(1)
          .padding_right(1)
          .padding_top(1)
          .padding_bottom(1)
          .kernel_height(3)
          .kernel_width(3)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, output_width_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, output_tuple_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__PSIMD, chw_layout) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(5)
        .output_width_stride(input_width)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC

#if !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_eq_4_pad0) {
    TEST_REQUIRES_PSIMD;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(0)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_lt_4_pad0) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_gt_4_pad0) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_div_4_pad0) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_stride_pad0) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_tuple_stride_pad0) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, output_height_gt_1_pad0) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_height = 2; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(1)
          .padding_right(1)
          .padding_top(0)
          .padding_bottom(1)
          .kernel_height(3)
          .kernel_width(3)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, output_width_stride_pad0) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, output_tuple_stride_pad0) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, chw_layout_pad0) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_eq_4_pad1) {
    TEST_REQUIRES_PSIMD;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_lt_4_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_gt_4_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_div_4_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_width_stride_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, input_tuple_stride_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, output_height_gt_1_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_height = 2; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(1)
          .padding_right(1)
          .padding_top(1)
          .padding_bottom(1)
          .kernel_height(3)
          .kernel_width(3)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, output_width_stride_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, output_tuple_stride_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__PSIMD, chw_layout_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC

#if !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, input_width_eq_4_pad2) {
    TEST_REQUIRES_PSIMD;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, input_width_lt_4_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, input_width_gt_4_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, input_width_div_4_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, input_width_stride_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, input_tuple_stride_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, output_height_eq_2_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(2)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, output_height_gt_2_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_height = 3; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(2)
          .padding_right(2)
          .padding_top(2)
        .padding_bottom(2)
          .kernel_height(5)
          .kernel_width(5)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, output_width_stride_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, output_tuple_stride_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__PSIMD, chw_layout_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      for (size_t output_height = 1; output_height < 32; output_height += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .input_width_stride(input_width)
          .padding_left(2)
          .padding_right(2)
          .padding_top(2)
          .padding_bottom(2)
          .kernel_height(5)
          .kernel_width(5)
          .output_height(5)
          .output_width_stride(input_width)
          .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
      }
    }
  }

#endif  // !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC

#if !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_eq_8_pad2) {
    TEST_REQUIRES_PSIMD;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(8)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_eq_8_pad1) {
    TEST_REQUIRES_PSIMD;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(8)
      .padding_left(2)
      .padding_right(2)
      .padding_top(1)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_lt_8_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_lt_8_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_gt_8_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 8; input_width < 16; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_gt_8_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 8; input_width < 16; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_div_4_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 16; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_div_4_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 16; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_stride_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_width_stride_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_tuple_stride_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, input_tuple_stride_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, output_height_gt_1_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_height = 3; output_height < 4; output_height++) {
      for (size_t input_width = 4; input_width < 5; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(2)
          .padding_right(2)
          .padding_top(1)
          .padding_bottom(2)
          .kernel_height(5)
          .kernel_width(5)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, output_height_gt_1_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t output_height = 3; output_height < 4; output_height++) {
      for (size_t input_width = 4; input_width < 5; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(2)
          .padding_right(2)
          .padding_top(2)
          .padding_bottom(2)
          .kernel_height(5)
          .kernel_width(5)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, output_width_stride_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, output_width_stride_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, output_tuple_stride_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, output_tuple_stride_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, chw_layout_pad1) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 1) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__PSIMD, chw_layout_pad2) {
    TEST_REQUIRES_PSIMD;
    for (size_t input_width = 1; input_width < 32; input_width += 1) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_eq_4_pad0) {
    TEST_REQUIRES_X86_SSE;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(0)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_lt_4_pad0) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_gt_4_pad0) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_div_4_pad0) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_stride_pad0) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_tuple_stride_pad0) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, output_height_gt_1_pad0) {
    TEST_REQUIRES_X86_SSE;
    for (size_t output_height = 2; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(1)
          .padding_right(1)
          .padding_top(0)
          .padding_bottom(1)
          .kernel_height(3)
          .kernel_width(3)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
      }
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, output_width_stride_pad0) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, output_tuple_stride_pad0) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, chw_layout_pad0) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_eq_4_pad1) {
    TEST_REQUIRES_X86_SSE;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_lt_4_pad1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_gt_4_pad1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_div_4_pad1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_width_stride_pad1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, input_tuple_stride_pad1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, output_height_gt_1_pad1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t output_height = 2; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(1)
          .padding_right(1)
          .padding_top(1)
          .padding_bottom(1)
          .kernel_height(3)
          .kernel_width(3)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
      }
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, output_width_stride_pad1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, output_tuple_stride_pad1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__SSE, chw_layout_pad1) {
    TEST_REQUIRES_X86_SSE;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM64
  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, input_width_eq_4_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
  }

  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, input_width_lt_4_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, input_width_gt_4_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, input_width_div_4_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, input_width_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, input_tuple_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, output_height_gt_1_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t output_height = 2; output_height <= 5; output_height++) {
      for (size_t input_width = 8; input_width < 9; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(1)
          .padding_right(1)
          .padding_top(1)
          .padding_bottom(1)
          .kernel_height(3)
          .kernel_width(3)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
      }
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, output_width_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, output_tuple_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3P1__NEONFMA, chw_layout_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(5)
        .output_width_stride(input_width)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_eq_4_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(0)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_eq_4_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_lt_4_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_lt_4_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_gt_4_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_gt_4_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_div_4_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_div_4_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_stride_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_width_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }


  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_tuple_stride_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, input_tuple_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, output_height_gt_1_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t output_height = 2; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(1)
          .padding_right(1)
          .padding_top(0)
          .padding_bottom(1)
          .kernel_height(3)
          .kernel_width(3)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
      }
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, output_height_gt_1_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t output_height = 2; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(1)
          .padding_right(1)
          .padding_top(1)
          .padding_bottom(1)
          .kernel_height(3)
          .kernel_width(3)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
      }
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, output_width_stride_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, output_width_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, output_tuple_stride_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, output_tuple_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, chw_layout_pad0) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_3X3S2P1__NEONFMA, chw_layout_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, input_width_eq_4_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(4)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, input_width_lt_4_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 4; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, input_width_gt_4_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 5; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, input_width_div_4_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 8; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, input_width_stride_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, input_tuple_stride_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, output_height_eq_2_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(2)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, output_height_gt_2_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t output_height = 3; output_height < 5; output_height++) {
      for (size_t input_width = 1; input_width < 32; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(2)
          .padding_right(2)
          .padding_top(2)
        .padding_bottom(2)
          .kernel_height(5)
          .kernel_width(5)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
      }
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, output_width_stride_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, output_tuple_stride_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5P2__NEONFMA, chw_layout_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      for (size_t output_height = 1; output_height < 32; output_height += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .input_width_stride(input_width)
          .padding_left(2)
          .padding_right(2)
          .padding_top(2)
          .padding_bottom(2)
          .kernel_height(5)
          .kernel_width(5)
          .output_height(5)
          .output_width_stride(input_width)
          .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_eq_8_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(8)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_eq_8_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    DWConvCHWMicrokernelTester()
      .input_tuple_size(4)
      .output_tuple_size(4)
      .input_width(8)
      .padding_left(2)
      .padding_right(2)
      .padding_top(1)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_lt_8_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_lt_8_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 8; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_gt_8_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 8; input_width < 16; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_gt_8_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 8; input_width < 16; input_width++) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_div_4_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 16; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_div_4_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 16; input_width < 32; input_width += 4) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_width_stride_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(36)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_tuple_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, input_tuple_stride_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 5) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(4)
        .input_tuple_stride(3 * 4)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, output_height_gt_1_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t output_height = 3; output_height < 4; output_height++) {
      for (size_t input_width = 4; input_width < 5; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(2)
          .padding_right(2)
          .padding_top(1)
          .padding_bottom(2)
          .kernel_height(5)
          .kernel_width(5)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
      }
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, output_height_gt_1_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t output_height = 3; output_height < 4; output_height++) {
      for (size_t input_width = 4; input_width < 5; input_width += 3) {
        DWConvCHWMicrokernelTester()
          .input_tuple_size(4)
          .output_tuple_size(4)
          .input_width(input_width)
          .padding_left(2)
          .padding_right(2)
          .padding_top(2)
          .padding_bottom(2)
          .kernel_height(5)
          .kernel_width(5)
          .subsampling(2)
          .output_height(output_height)
          .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
      }
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, output_width_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, output_width_stride_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(36)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, output_tuple_stride_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, output_tuple_stride_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride(4)
        .output_tuple_stride(5 * 4)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, chw_layout_pad1) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 1) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(1)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }

  TEST(F32_DWCONV_CHW_5X5S2P2__NEONFMA, chw_layout_pad2) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t input_width = 1; input_width < 32; input_width += 1) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(4)
        .output_tuple_size(4)
        .input_width(input_width)
        .input_width_stride(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .subsampling(2)
        .output_height(5)
        .output_width_stride((input_width - 1) / 2 + 1)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma);
    }
  }
#endif  // XNN_ARCH_ARM64


TEST(F32_DWCONV_CHW_3X3P1__SCALAR, input_width_eq_1) {
  DWConvCHWMicrokernelTester()
    .input_tuple_size(1)
    .output_tuple_size(1)
    .input_width(1)
    .padding_left(1)
    .padding_right(1)
    .padding_top(1)
    .padding_bottom(1)
    .kernel_height(3)
    .kernel_width(3)
    .output_height(1)
    .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
}

TEST(F32_DWCONV_CHW_3X3P1__SCALAR, input_width_gt_1) {
  for (size_t input_width = 2; input_width < 32; input_width++) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3P1__SCALAR, input_width_stride) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(36)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3P1__SCALAR, input_tuple_stride) {
  for (size_t input_width = 1; input_width < 32; input_width += 5) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(4)
      .input_tuple_stride(3 * 4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3P1__SCALAR, output_height_gt_1) {
  for (size_t output_height = 2; output_height < 5; output_height++) {
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(1)
        .output_tuple_size(1)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .output_height(output_height)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_DWCONV_CHW_3X3P1__SCALAR, output_width_stride) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .output_height(5)
      .output_width_stride(36)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3P1__SCALAR, output_tuple_stride) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .output_height(5)
      .output_width_stride(4)
      .output_tuple_stride(5 * 4)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3P1__SCALAR, chw_layout) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .output_height(5)
      .output_width_stride(input_width)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, input_width_eq_1_pad0) {
  DWConvCHWMicrokernelTester()
    .input_tuple_size(1)
    .output_tuple_size(1)
    .input_width(1)
    .padding_left(1)
    .padding_right(1)
    .padding_top(0)
    .padding_bottom(1)
    .kernel_height(3)
    .kernel_width(3)
    .subsampling(2)
    .output_height(1)
    .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, input_width_gt_1_pad0) {
  for (size_t input_width = 2; input_width < 32; input_width++) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(0)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, input_width_stride_pad0) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(36)
      .padding_left(1)
      .padding_right(1)
      .padding_top(0)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, input_tuple_stride_pad0) {
  for (size_t input_width = 1; input_width < 32; input_width += 5) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(4)
      .input_tuple_stride(3 * 4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(0)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, output_height_gt_1_pad0) {
  for (size_t output_height = 2; output_height < 5; output_height++) {
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(1)
        .output_tuple_size(1)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(0)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(output_height)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, output_width_stride_pad0) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(0)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(5)
      .output_width_stride(36)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, output_tuple_stride_pad0) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(0)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(5)
      .output_width_stride(4)
      .output_tuple_stride(5 * 4)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, chw_layout_pad0) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(0)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(5)
      .output_width_stride(input_width)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, input_width_eq_1_pad1) {
  DWConvCHWMicrokernelTester()
    .input_tuple_size(1)
    .output_tuple_size(1)
    .input_width(1)
    .padding_left(1)
    .padding_right(1)
    .padding_top(1)
    .padding_bottom(1)
    .kernel_height(3)
    .kernel_width(3)
    .subsampling(2)
    .output_height(1)
    .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, input_width_gt_1_pad1) {
  for (size_t input_width = 2; input_width < 32; input_width++) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, input_width_stride_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(36)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, input_tuple_stride_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 5) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(4)
      .input_tuple_stride(3 * 4)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, output_height_gt_1_pad1) {
  for (size_t output_height = 2; output_height < 5; output_height++) {
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(1)
        .output_tuple_size(1)
        .input_width(input_width)
        .padding_left(1)
        .padding_right(1)
        .padding_top(1)
        .padding_bottom(1)
        .kernel_height(3)
        .kernel_width(3)
        .subsampling(2)
        .output_height(output_height)
        .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, output_width_stride_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(5)
      .output_width_stride(36)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, output_tuple_stride_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(5)
      .output_width_stride(4)
      .output_tuple_stride(5 * 4)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_3X3S2P1__SCALAR, chw_layout_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(input_width)
      .padding_left(1)
      .padding_right(1)
      .padding_top(1)
      .padding_bottom(1)
      .kernel_height(3)
      .kernel_width(3)
      .subsampling(2)
      .output_height(5)
      .output_width_stride(input_width)
      .Test(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5P2__SCALAR, input_width_eq_1_pad2) {
  DWConvCHWMicrokernelTester()
    .input_tuple_size(1)
    .output_tuple_size(1)
    .input_width(1)
    .padding_left(2)
    .padding_right(2)
    .padding_top(2)
    .padding_bottom(2)
    .kernel_height(5)
    .kernel_width(5)
    .output_height(1)
    .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
}

TEST(F32_DWCONV_CHW_5X5P2__SCALAR, input_width_gt_1_pad2) {
  for (size_t input_width = 2; input_width < 32; input_width++) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5P2__SCALAR, input_width_stride_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(36)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5P2__SCALAR, input_tuple_stride_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 5) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(4)
      .input_tuple_stride(3 * 4)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5P2__SCALAR, output_height_gt_1_pad2) {
  for (size_t output_height = 2; output_height < 5; output_height++) {
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(1)
        .output_tuple_size(1)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
      .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(output_height)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_DWCONV_CHW_5X5P2__SCALAR, output_width_stride_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(5)
      .output_width_stride(36)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5P2__SCALAR, output_tuple_stride_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(5)
      .output_width_stride(4)
      .output_tuple_stride(5 * 4)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5P2__SCALAR, chw_layout_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(5)
      .output_width_stride(input_width)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, input_width_eq_1_pad1) {
  DWConvCHWMicrokernelTester()
    .input_tuple_size(1)
    .output_tuple_size(1)
    .input_width(1)
    .padding_left(2)
    .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
    .kernel_height(5)
    .kernel_width(5)
    .output_height(1)
    .subsampling(2)
    .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, input_width_gt_1_pad1) {
  for (size_t input_width = 2; input_width < 32; input_width++) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, input_width_stride_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(36)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, input_tuple_stride_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 5) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(4)
      .input_tuple_stride(3 * 4)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, output_height_gt_1_pad1) {
  for (size_t output_height = 2; output_height < 5; output_height++) {
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(1)
        .output_tuple_size(1)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(output_height)
        .subsampling(2)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, output_width_stride_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(5)
      .output_width_stride(36)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, output_tuple_stride_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(5)
      .output_width_stride(4)
      .output_tuple_stride(5 * 4)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, chw_layout_pad1) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(5)
      .output_width_stride(input_width)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, input_width_eq_1_pad2) {
  DWConvCHWMicrokernelTester()
    .input_tuple_size(1)
    .output_tuple_size(1)
    .input_width(1)
    .padding_left(2)
    .padding_right(2)
    .padding_top(2)
    .padding_bottom(2)
    .kernel_height(5)
    .kernel_width(5)
    .output_height(1)
    .subsampling(2)
    .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, input_width_gt_1_pad2) {
  for (size_t input_width = 2; input_width < 32; input_width++) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, input_width_stride_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(36)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, input_tuple_stride_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 5) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(4)
      .input_tuple_stride(3 * 4)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(1)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, output_height_gt_1_pad2) {
  for (size_t output_height = 2; output_height < 5; output_height++) {
    for (size_t input_width = 1; input_width < 32; input_width += 3) {
      DWConvCHWMicrokernelTester()
        .input_tuple_size(1)
        .output_tuple_size(1)
        .input_width(input_width)
        .padding_left(2)
        .padding_right(2)
        .padding_top(2)
        .padding_bottom(2)
        .kernel_height(5)
        .kernel_width(5)
        .output_height(output_height)
        .subsampling(2)
        .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, output_width_stride_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(5)
      .output_width_stride(36)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, output_tuple_stride_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(5)
      .output_width_stride(4)
      .output_tuple_stride(5 * 4)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_DWCONV_CHW_5X5S2P2__SCALAR, chw_layout_pad2) {
  for (size_t input_width = 1; input_width < 32; input_width += 3) {
    DWConvCHWMicrokernelTester()
      .input_tuple_size(1)
      .output_tuple_size(1)
      .input_width(input_width)
      .input_width_stride(input_width)
      .padding_left(2)
      .padding_right(2)
      .padding_top(2)
      .padding_bottom(2)
      .kernel_height(5)
      .kernel_width(5)
      .output_height(5)
      .output_width_stride(input_width)
      .subsampling(2)
      .Test(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar, DWConvCHWMicrokernelTester::Variant::Scalar);
  }
}
