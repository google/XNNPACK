// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-gemm-minmax-rndnu.yaml
//   Generator: tools/generate-gemm-test.py


#include <gtest/gtest.h>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8__ASM_AARCH32_NEON_MLAL_LANE_CORTEX_A7, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2__NEON_MULL_LD4R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2__neon_mull_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(1)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C2S4__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(1)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X8C4__NEON_MULL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(1)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2__NEON_MLAL_LD4R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2__neon_mlal_ld4r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(1)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(1)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(1)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(2)
        .sr(4)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(2)
          .sr(4)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(1)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(1)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C2S4__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(2)
      .sr(4)
      .m(1)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_DUP, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__NEON_MLAL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4S2__NEON_MULL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(1)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C2__NEON_MULL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4__NEON_MULL_DUP, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(2)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(2)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(2)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(2)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(2)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(4)
        .sr(2)
        .m(2)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(8)
          .kr(4)
          .sr(2)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(8)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(2)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(2)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X8C4S2__NEON_MULL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(8)
      .kr(4)
      .sr(2)
      .m(2)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x8c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C2__NEON_MULL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C2__NEON_MLAL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8C4__NEON_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(3)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8c4__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MLAL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mlal_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C2__NEON_MULL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c2__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MLAL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16C4__NEON_MULL_DUP, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2__NEON_MLAL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(2)
        .sr(4)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(2)
          .sr(4)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(2)
            .sr(4)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C2S4__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(2)
      .sr(4)
      .m(4)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c2s4__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MLAL_DUP, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mlal_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(4)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C4__NEON_MULL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(4)
      .sr(1)
      .m(4)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c4__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(2)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(2)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(2)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C2__NEON_MULL_LD2R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(2)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c2__neon_mull_ld2r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MLAL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mlal_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_DUP, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__NEON_MULL_LD1R, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__neon_mull_ld1r, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(4)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(2)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(2)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(2)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(4)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(4)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4S2__NEON_MULL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(2)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4s2__neon_mull, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .a_stride(7)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_div_4) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(43)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD32, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_eq_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 1; m++) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 1; m++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_lt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_gt_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_div_8) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(1)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(1)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(1)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(1)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 1; m++) {
          GemmMicrokernelTester()
            .mr(1)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_1X16C4__ASM_AARCH64_NEONDOT_LD64, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(1)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(1)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_1x16c4__asm_aarch64_neondot_ld64, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .a_stride(7)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(m)
          .n(n)
          .k(4)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4_subtile_m) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(m)
        .n(16)
        .k(4)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_eq_4_subtile_n) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(n)
        .k(4)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_lt_4) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_lt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 4; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k < 4; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_gt_4) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_gt_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 5; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 5; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_div_4) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_div_4_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 8; k <= 40; k += 4) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(4)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(43)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, k_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 8; k <= 40; k += 4) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, n_gt_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, n_div_16) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(4)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(23)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 20; k += 5) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON_DOT;
    for (size_t k = 1; k <= 20; k += 5) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(4)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, qmin) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, qmax) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16C4__ASM_AARCH64_NEONDOT_LD32, strided_cm) {
    TEST_REQUIRES_ARM_NEON_DOT;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(4)
      .sr(1)
      .m(4)
      .n(16)
      .k(4)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld32, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16__NEON_MLAL_LANE_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(2)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(16)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(16)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(16)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(16)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(m)
        .n(16)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(2)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(16)
        .kr(16)
        .sr(1)
        .m(2)
        .n(16)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(16)
          .kr(16)
          .sr(1)
          .m(2)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 2; m++) {
          GemmMicrokernelTester()
            .mr(2)
            .nr(16)
            .kr(16)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(16)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(16)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_2X16C16__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(2)
      .nr(16)
      .kr(16)
      .sr(1)
      .m(2)
      .n(16)
      .k(16)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x16c16__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(3)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X8__NEON_MULL_ADDW_DUP, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(3)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x8__neon_mull_addw_dup, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(3)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(3)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 3; m++) {
          GemmMicrokernelTester()
            .mr(3)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_3X16__NEON_MLAL_LANE_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(3)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(3)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_eq_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .a_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_eq_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(m)
          .n(n)
          .k(16)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_eq_16_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(m)
        .n(8)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_eq_16_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(n)
        .k(16)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_lt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_lt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(37)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 17; k < 32; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(8)
        .kr(8)
        .sr(1)
        .m(4)
        .n(8)
        .k(k)
        .a_stride(163)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, k_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 32; k <= 160; k += 16) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(8)
          .kr(8)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(83)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 80; k += 17) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 80; k += 17) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(8)
            .kr(8)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X8C8__NEON_MLAL, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(8)
      .kr(8)
      .sr(1)
      .m(4)
      .n(8)
      .k(16)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x8c8__neon_mlal, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cn_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(m)
        .n(16)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 16; n++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(16)
        .kr(1)
        .sr(1)
        .m(4)
        .n(16)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_gt_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 17; n < 32; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .cn_stride(19)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(16)
          .kr(1)
          .sr(1)
          .m(4)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, n_div_16_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 32; n <= 48; n += 16) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 16; n++) {
        for (uint32_t m = 1; m <= 4; m++) {
          GemmMicrokernelTester()
            .mr(4)
            .nr(16)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(19)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_4X16__NEON_MLAL_LANE, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(4)
      .nr(16)
      .kr(1)
      .sr(1)
      .m(4)
      .n(16)
      .k(8)
      .cm_stride(19)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cn_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_eq_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      for (uint32_t m = 1; m <= 6; m++) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(8)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_m) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t m = 1; m <= 6; m++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(m)
        .n(8)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_eq_8_subtile_n) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 1; n <= 8; n++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(n)
        .k(8)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_lt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(11)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k < 8; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(19)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 9; k < 16; k++) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      GemmMicrokernelTester()
        .mr(6)
        .nr(8)
        .kr(1)
        .sr(1)
        .m(6)
        .n(8)
        .k(k)
        .a_stride(83)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, k_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 16; k <= 80; k += 8) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_gt_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_gt_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 9; n < 16; n++) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_div_8_strided_cn) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .cn_stride(11)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_div_8_strided_a) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        GemmMicrokernelTester()
          .mr(6)
          .nr(8)
          .kr(1)
          .sr(1)
          .m(6)
          .n(n)
          .k(k)
          .a_stride(43)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t n = 16; n <= 24; n += 8) {
      for (size_t k = 1; k <= 40; k += 9) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, strided_cm_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t k = 1; k <= 40; k += 9) {
      for (uint32_t n = 1; n <= 8; n++) {
        for (uint32_t m = 1; m <= 6; m++) {
          GemmMicrokernelTester()
            .mr(6)
            .nr(8)
            .kr(1)
            .sr(1)
            .m(m)
            .n(n)
            .k(k)
            .cm_stride(11)
            .iterations(1)
            .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
        }
      }
    }
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, qmin) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmin(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, qmax) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .qmax(128)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }

  TEST(QS8_GEMM_MINMAX_RNDNU_6X8__NEON_MLAL_LANE_PRFM, strided_cm) {
    TEST_REQUIRES_ARM_NEON;
    GemmMicrokernelTester()
      .mr(6)
      .nr(8)
      .kr(1)
      .sr(1)
      .m(6)
      .n(8)
      .k(8)
      .cm_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane_prfm, xnn_init_qs8_conv_minmax_rndnu_neon_params, xnn_qs8_requantize_rndnu);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .cn_stride(5)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 2; m++) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 2; m++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, k_gt_1_strided_a) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(2)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(2)
      .n(2)
      .k(k)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, n_gt_2_strided_a) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, n_div_2_strided_a) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(2)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 2; m++) {
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .qmin(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .qmax(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_2X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(2)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(2)
    .n(2)
    .k(1)
    .cm_stride(5)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_2x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}


TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(1)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(1)
    .cn_stride(5)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 3; m++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(3)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(3)
      .n(2)
      .k(k)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, k_gt_1_strided_a) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(3)
      .n(2)
      .k(k)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, n_gt_2_strided_a) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, n_div_2_strided_a) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(1)
    .qmin(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(1)
    .qmax(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(3)
    .n(2)
    .k(1)
    .cm_stride(5)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}


TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 3; m++) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 3; m++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(3)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(3)
      .n(4)
      .k(k)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, k_gt_1_strided_a) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(3)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(3)
      .n(4)
      .k(k)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(3)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(3)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 3; m++) {
        GemmMicrokernelTester()
          .mr(3)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .qmin(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .qmax(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_3X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(3)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(3)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_3x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}


TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cn_stride(5)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 2; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(m)
      .n(2)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 2; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, k_gt_1_strided_a) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(2)
      .kr(1)
      .sr(1)
      .m(4)
      .n(2)
      .k(k)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, n_gt_2) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, n_gt_2_strided_cn) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, n_gt_2_strided_a) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, n_gt_2_subtile) {
  for (uint32_t n = 3; n < 4; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, n_div_2) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, n_div_2_strided_cn) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(5)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, n_div_2_strided_a) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(2)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, n_div_2_subtile) {
  for (uint32_t n = 4; n <= 6; n += 2) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 2; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(2)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(5)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .qmin(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .qmax(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X2__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(2)
    .kr(1)
    .sr(1)
    .m(4)
    .n(2)
    .k(1)
    .cm_stride(5)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x2__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}


TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, k_eq_1) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, strided_cn) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cn_stride(7)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, k_eq_1_strided_a) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .a_stride(3)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, k_eq_1_subtile) {
  for (uint32_t n = 1; n <= 4; n++) {
    for (uint32_t m = 1; m <= 4; m++) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(m)
        .n(n)
        .k(1)
        .iterations(1)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, k_eq_1_subtile_m) {
  for (uint32_t m = 1; m <= 4; m++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(m)
      .n(4)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, k_eq_1_subtile_n) {
  for (uint32_t n = 1; n <= 4; n++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(n)
      .k(1)
      .iterations(1)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, k_gt_1) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, k_gt_1_strided_a) {
  for (size_t k = 2; k < 10; k++) {
    GemmMicrokernelTester()
      .mr(4)
      .nr(4)
      .kr(1)
      .sr(1)
      .m(4)
      .n(4)
      .k(k)
      .a_stride(11)
      .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, k_gt_1_subtile) {
  for (size_t k = 2; k < 10; k++) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, n_gt_4) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, n_gt_4_strided_cn) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, n_gt_4_strided_a) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, n_gt_4_subtile) {
  for (uint32_t n = 5; n < 8; n++) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, n_div_4) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, n_div_4_strided_cn) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .cn_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, n_div_4_strided_a) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(1)
        .sr(1)
        .m(4)
        .n(n)
        .k(k)
        .a_stride(7)
        .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, n_div_4_subtile) {
  for (uint32_t n = 8; n <= 12; n += 4) {
    for (size_t k = 1; k <= 5; k += 2) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, strided_cm_subtile) {
  for (size_t k = 1; k <= 5; k += 2) {
    for (uint32_t n = 1; n <= 4; n++) {
      for (uint32_t m = 1; m <= 4; m++) {
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(1)
          .sr(1)
          .m(m)
          .n(n)
          .k(k)
          .cm_stride(7)
          .iterations(1)
          .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
      }
    }
  }
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, qmin) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .qmin(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, qmax) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .qmax(128)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}

TEST(QS8_GEMM_MINMAX_RNDNU_4X4__SCALAR, strided_cm) {
  GemmMicrokernelTester()
    .mr(4)
    .nr(4)
    .kr(1)
    .sr(1)
    .m(4)
    .n(4)
    .k(1)
    .cm_stride(7)
    .Test(xnn_qs8_gemm_minmax_rndnu_ukernel_4x4__scalar, xnn_init_qs8_conv_minmax_rndnu_scalar_params, xnn_qs8_requantize_rndnu);
}
