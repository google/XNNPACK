// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-vmulc-minmax-rndnu.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"
#include "vbinaryc-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u8, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD64_U16, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld64_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
      }
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }

  TEST(QU8_VMULC_MINMAX_RNDNU__NEON_LD128_U16, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qu8_vmulc_minmax_rndnu_ukernel__neon_ld128_u16, xnn_init_qu8_mul_minmax_rndnu_neon_params, xnn_qu8_requantize_rndnu);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
