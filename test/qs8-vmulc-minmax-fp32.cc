// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-vmulc-minmax-fp32.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"
#include "vbinaryc-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD64_U16, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEON_LD128_U16, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neon_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, a_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, b_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, y_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, a_scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, b_scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, y_scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U8, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, a_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, b_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, y_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, a_scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, b_scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, y_scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD64_U16, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, a_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, b_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, y_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, a_scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, b_scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, y_scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__NEONV8_LD128_U16, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__neonv8_ld128_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, a_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, b_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, a_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, b_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U8, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, a_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, b_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, a_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, b_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE2_MUL16_LD64_U16, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse2_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U8, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__SSE41_MUL16_LD64_U16, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__sse41_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, a_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, b_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, y_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, a_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, b_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, y_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U8, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, a_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, b_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, y_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, a_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, b_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, y_scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__AVX_MUL16_LD64_U16, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__avx_mul16_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, batch_eq_8) {
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, a_zero_point) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, b_zero_point) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, y_zero_point) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, a_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, b_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, y_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u8, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, batch_eq_16) {
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, a_zero_point) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, b_zero_point) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, y_zero_point) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, a_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, b_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, y_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, qmin) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__WASMSIMD_MUL32_LD64_U16, qmax) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__wasmsimd_mul32_ld64_u16, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, batch_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VBinaryCMicrokernelTester()
      .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t))
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, batch_div_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 2 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size < 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, batch_lt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size < 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, batch_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 2 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size < 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 2) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, a_zero_point) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, b_zero_point) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, y_zero_point) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, a_scale) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, b_scale) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, y_scale) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U1V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u1v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, batch_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VBinaryCMicrokernelTester()
      .batch_size(2 * xnn_init_hardware_config()->vlenb / sizeof(int8_t))
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, batch_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 4 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size < 20 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 2 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, batch_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size < 2 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, batch_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 3 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 4) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, a_zero_point) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, b_zero_point) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, y_zero_point) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, a_scale) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, b_scale) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, y_scale) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
      }
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }

  TEST(QS8_VMULC_MINMAX_FP32__RVV_U2V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(int8_t);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(int8_t)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, batch_eq_1) {
  VBinaryCMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, a_zero_point) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .a_zero_point(a_zero_point)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, b_zero_point) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .b_zero_point(b_zero_point)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, y_zero_point) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .y_zero_point(y_zero_point)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, a_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .a_scale(a_scale)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, b_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .b_scale(b_scale)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, y_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .y_scale(y_scale)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, qmin) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U1, qmax) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u1, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, batch_eq_2) {
  VBinaryCMicrokernelTester()
    .batch_size(2)
    .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, a_zero_point) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .a_zero_point(a_zero_point)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, b_zero_point) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .b_zero_point(b_zero_point)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, y_zero_point) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .y_zero_point(y_zero_point)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, a_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .a_scale(a_scale)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, b_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .b_scale(b_scale)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, y_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .y_scale(y_scale)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, qmin) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U2, qmax) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, batch_eq_4) {
  VBinaryCMicrokernelTester()
    .batch_size(4)
    .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, a_zero_point) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .a_zero_point(a_zero_point)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, b_zero_point) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .b_zero_point(b_zero_point)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, y_zero_point) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .y_zero_point(y_zero_point)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, a_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .a_scale(a_scale)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, b_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .b_scale(b_scale)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, y_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .y_scale(y_scale)
        .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
    }
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, qmin) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}

TEST(QS8_VMULC_MINMAX_FP32__SCALAR_U4, qmax) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u4, xnn_init_qs8_mul_minmax_scalar_params, xnn_qs8_requantize_fp32);
  }
}