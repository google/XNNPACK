// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vmulcaddc-minmax.yaml
//   Generator: tools/generate-vmulcaddc-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vmulcaddc.h>
#include "vmulcaddc-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VMulCAddCMicrokernelTester()
      .channel_tile(8)
      .channels(8)
      .rows(2)
      .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, channels_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 16; channels < 80; channels += 8) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 9; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, rows_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .input_stride(43)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .output_stride(43)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__NEONFP16ARITH_2X, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VMulCAddCMicrokernelTester()
      .channel_tile(16)
      .channels(16)
      .rows(2)
      .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, channels_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 32; channels < 160; channels += 16) {
      VMulCAddCMicrokernelTester()
        .channel_tile(16)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(16)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 17; channels < 32; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(16)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, rows_div_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, input_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .input_stride(83)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, output_stride) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .output_stride(83)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__NEONFP16ARITH_2X, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__neonfp16arith_2x, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, channels_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VMulCAddCMicrokernelTester()
      .channel_tile(8)
      .channels(8)
      .rows(2)
      .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, channels_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 16; channels < 80; channels += 8) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, channels_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels < 8; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, channels_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 9; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(8)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, rows_lt_2) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, rows_div_2) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, rows_gt_2) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, input_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .input_stride(43)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .output_stride(43)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C8__FMA3_2X, qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        VMulCAddCMicrokernelTester()
          .channel_tile(8)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, channels_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VMulCAddCMicrokernelTester()
      .channel_tile(16)
      .channels(16)
      .rows(2)
      .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, channels_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 32; channels < 160; channels += 16) {
      VMulCAddCMicrokernelTester()
        .channel_tile(16)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, channels_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 1; channels < 16; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(16)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, channels_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t channels = 17; channels < 32; channels++) {
      VMulCAddCMicrokernelTester()
        .channel_tile(16)
        .channels(channels)
        .rows(2)
        .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, rows_lt_2) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, rows_div_2) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, rows_gt_2) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, input_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .input_stride(83)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, output_stride) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .output_stride(83)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .inplace(true)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, qmin) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .qmin(128)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VMULCADDC_MINMAX_C16__FMA3_2X, qmax) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        VMulCAddCMicrokernelTester()
          .channel_tile(16)
          .channels(channels)
          .rows(rows)
          .qmax(128)
          .Test(xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
